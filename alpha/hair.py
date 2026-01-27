import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# GPU HARD REQUIREMENT
# ==========================================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA REQUIRED. GPU NOT FOUND.")

DEVICE = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

# ==========================================================
# SILENCE YOLO LOGS
# ==========================================================
LOGGER.setLevel("ERROR")

# ==========================================================
# CONFIG
# ==========================================================
GALLERY_VIDEO = "cam1.mp4"
QUERY_VIDEO   = "cam2.mp4"

SIM_THRESHOLD = 0.35
MAX_TARGET_SAMPLES = 50

# ==========================================================
# LOAD MODELS
# ==========================================================
print("[INFO] Loading YOLO models...")
person_model = YOLO("yolov8x.pt")
pose_model   = YOLO("yolov8x-pose.pt")

print("[INFO] Loading ResNet-18...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE).eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# CLICK SELECTION
# ==========================================================
selected_bbox = None
clicked = False

def mouse_callback(event, x, y, flags, boxes):
    global selected_bbox, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        for (x1,y1,x2,y2) in boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_bbox = (x1,y1,x2,y2)
                clicked = True
                break

# ==========================================================
# DRAW
# ==========================================================
def draw_box(frame, box, color, label=None):
    x1,y1,x2,y2 = box
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ==========================================================
# STRICT HAIR EXTRACTION (NO FACE LEAKAGE)
# ==========================================================
def extract_hair(person_crop):
    h, w, _ = person_crop.shape

    pose = pose_model(person_crop, verbose=False)[0]

    if pose.keypoints is not None:
        kps = pose.keypoints.xy
        if kps is not None and len(kps) > 0:
            kp = kps.cpu().numpy()[0]
            nose, leye, reye = kp[0], kp[1], kp[2]

            if not (np.any(nose == 0) or np.any(leye == 0) or np.any(reye == 0)):
                eye_y = min(leye[1], reye[1])
                nose_y = nose[1]
                head_h = abs(eye_y - nose_y)

                width = int(abs(leye[0] - reye[0]) * 2)
                x_center = int((leye[0] + reye[0]) / 2)

                y1 = max(0, int(eye_y - 2 * head_h))
                y2 = max(0, int(eye_y - head_h * 0.3))  # REMOVE FACE
                x1 = max(0, x_center - width // 2)
                x2 = min(w, x_center + width // 2)

                hair = person_crop[y1:y2, x1:x2]

                if hair.size > 0 and hair.shape[0] > 40:
                    return hair

    # fallback: STRICT top region
    fallback = person_crop[: int(0.20 * h), :]
    return fallback if fallback.shape[0] > 40 else None

# ==========================================================
# EMBEDDING (WHITENED)
# ==========================================================
@torch.no_grad()
def embed_hair(hair):
    img = transform(hair).unsqueeze(0).to(DEVICE)
    emb = resnet(img).cpu().numpy()[0]

    # WHITENING (CRITICAL)
    emb = emb - np.mean(emb)
    emb = emb / np.linalg.norm(emb)

    return emb

def aggregate_embeddings(embs):
    emb = np.mean(embs, axis=0)
    emb = emb - np.mean(emb)
    return emb / np.linalg.norm(emb)

# ==========================================================
# STEP 1 — CLICK TARGET
# ==========================================================
cap = cv2.VideoCapture(GALLERY_VIDEO)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read gallery video")

res = person_model(frame, classes=[0], verbose=False)[0]
boxes = []

for box in res.boxes.xyxy:
    x1,y1,x2,y2 = map(int, box)
    boxes.append((x1,y1,x2,y2))
    draw_box(frame, (x1,y1,x2,y2), (0,255,0))

cv2.imshow("CLICK TARGET PERSON", frame)
cv2.setMouseCallback("CLICK TARGET PERSON", mouse_callback, boxes)

while not clicked:
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()

print("[INFO] Target selected")

# ==========================================================
# STEP 2 — BUILD TARGET HAIR EMBEDDING (CLEAN)
# ==========================================================
target_embs = []
cap = cv2.VideoCapture(GALLERY_VIDEO)

while cap.isOpened() and len(target_embs) < MAX_TARGET_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    x1,y1,x2,y2 = selected_bbox
    crop = frame[y1:y2, x1:x2]
    hair = extract_hair(crop)

    if hair is not None:
        target_embs.append(embed_hair(hair))

cap.release()

TARGET_EMB = aggregate_embeddings(target_embs)
print(f"[INFO] Target samples used: {len(target_embs)}")

# ==========================================================
# STEP 3 — SIMILARITY-LOCKED VISUAL TRACKING
# ==========================================================
cap = cv2.VideoCapture(GALLERY_VIDEO)
print("[INFO] Running similarity-based tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = person_model(frame, classes=[0], verbose=False)[0]

    best_score = -1
    best_box = None

    for box in res.boxes.xyxy:
        x1,y1,x2,y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        hair = extract_hair(crop)
        if hair is None:
            continue

        emb = embed_hair(hair)
        score = cosine_similarity(
            TARGET_EMB.reshape(1,-1),
            emb.reshape(1,-1)
        )[0][0]

        draw_box(frame, (x1,y1,x2,y2), (0,0,255), f"{score:.2f}")

        if score > best_score:
            best_score = score
            best_box = (x1,y1,x2,y2)

    if best_box and best_score > SIM_THRESHOLD:
        draw_box(frame, best_box, (0,255,0), f"TARGET {best_score:.2f}")

    cv2.imshow("HAIR-ONLY SIMILARITY LOCK", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[DONE] Hair-only ReID finished")
