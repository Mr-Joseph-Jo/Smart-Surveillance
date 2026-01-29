import cv2
from ultralytics import YOLO

# Load YOLO model (use your custom model path if needed)
model = YOLO("trainedmodel.pt")  # or "best.pt"

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame, conf=0.6)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
