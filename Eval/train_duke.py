import torchreid
import torch
import os
import os.path as osp
import glob
from torchreid.data import ImageDataset

# ==================== 1. DEFINE GENERIC DATASET CLASS ====================
class DukePartDataset(ImageDataset):
    """
    A generic dataset class that works for ANY Duke-like folder structure.
    It works for 'upperbody', 'lowerbody', 'masked', or 'hair' 
    as long as the root path is correct.
    """
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        
        # Standard DukeMTMC Subfolders
        train_dir = osp.join(self.root, 'bounding_box_train')
        query_dir = osp.join(self.root, 'query')
        gallery_dir = osp.join(self.root, 'bounding_box_test')
        
        # Validation: Stop immediately if folders are missing
        if not osp.exists(train_dir):
            raise RuntimeError(f"Dataset not found at '{train_dir}'. Did you run create_part_crops.py?")

        # Load and Relabel Data
        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)

        super(DukePartDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []
        
        for img_path in img_paths:
            filename = osp.basename(img_path)
            # Decipher Duke Filename: 0001_c1_f0053048.jpg
            parts = filename.split('_')
            pid = int(parts[0])
            camid = int(parts[1].replace('c', ''))
            data.append((img_path, pid, camid))
        
        if relabel:
            return self.relabel_dataset(data)
        return data

    def relabel_dataset(self, dataset):
        # Maps disjoint PIDs (1, 5, 10) to continuous labels (0, 1, 2)
        pids = sorted(list(set([pid for _, pid, _ in dataset])))
        pid2label = {pid: i for i, pid in enumerate(pids)}
        new_dataset = []
        for img_path, pid, camid in dataset:
            new_dataset.append((img_path, pid2label[pid], camid))
        return new_dataset

# Register the class under a generic name
torchreid.data.register_image_dataset('duke_custom', DukePartDataset)

# ==================== 2. TRAINING FUNCTION ====================

def train_part(part_name, root_path):
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING FOR: {part_name.upper()} EXPERT")
    print(f"Data Root: {root_path}")
    print(f"{'='*60}")

    # 1. Configuration
    MAX_EPOCHS = 350       # 60 is the sweet spot for specialized parts
    BATCH_SIZE = 128       # High batch size stabilizes BatchNorm
    SAVE_DIR = f'log/osnet_duke_{part_name}'

    # 2. Data Manager
    # We pass the specific ROOT path for this part
    datamanager = torchreid.data.ImageDataManager(
        root=root_path,
        sources='duke_custom',  # Calls DukePartDataset(root=root_path)
        targets='duke_custom',
        height=128,             # Parts are smaller, so 128x128 is perfect
        width=128,              # Square aspect ratio for parts works best
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        transforms=['random_flip', 'random_erase', 'color_jitter'] # Strong augmentation
    )

    print(f"Building Model: OSNet (AIN)...")

    # 3. Build Model
    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True 
    )

    model = model.cuda()

    # 4. Optimizer & Scheduler
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=30 
    )

    # 5. Engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # 6. Run
    print(f"Training {part_name}... Logs will be saved to {SAVE_DIR}")
    engine.run(
        save_dir=SAVE_DIR,
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=5,
        test_only=False,
        fixbase_epoch=3 # Freeze base layers briefly to adapt to new part domain
    )

# ==================== 3. MAIN EXECUTION ====================

if __name__ == '__main__':
    # Train Upper Body Expert
    train_part(
        part_name='upper', 
        root_path='./duke_patches/upperbody'
    )
    
    # Train Lower Body Expert
    train_part(
        part_name='lower', 
        root_path='./duke_patches/lowerbody'
    )