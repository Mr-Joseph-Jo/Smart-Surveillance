import torchreid
import torch
import os.path as osp
import glob
from torchreid.data import ImageDataset

# ==================== 1. DEFINE CUSTOM MASKED DATASET ====================
class DukeMasked(ImageDataset):
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        
        # Point to the folders created by prepare_maskeddataset.py
        # Note: We assume root points to ".../DukeMTMC-Masked"
        train_dir = osp.join(self.root, 'bounding_box_train')
        query_dir = osp.join(self.root, 'query')
        gallery_dir = osp.join(self.root, 'bounding_box_test')
        
        # Safety Check
        if not osp.exists(train_dir):
            raise RuntimeError(f"'{train_dir}' does not exist. Did you run prepare_maskeddataset.py?")

        # Load and Relabel Data
        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)

        super(DukeMasked, self).__init__(train, query, gallery, **kwargs)

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
        pids = sorted(list(set([pid for _, pid, _ in dataset])))
        pid2label = {pid: i for i, pid in enumerate(pids)}
        new_dataset = []
        for img_path, pid, camid in dataset:
            new_dataset.append((img_path, pid2label[pid], camid))
        return new_dataset

# Register it so DataManager can find it
torchreid.data.register_image_dataset('duke_masked', DukeMasked)

# ==================== 2. MAIN TRAINING LOOP ====================

def main():
    # --- CONFIGURATION ---
    # Point this to the OUTPUT folder from your previous script
    MASKED_DATA_PATH = "./duke-Masked" 
    MAX_EPOCHS = 60       # Increased slightly for better convergence on clean data
    BATCH_SIZE = 64       # Increased to 64 (Better for Batch Normalization)
    
    print(f"Initializing Data Manager for Duke Masked Dataset...")

    # 3. Data Manager
    datamanager = torchreid.data.ImageDataManager(
        root=MASKED_DATA_PATH,
        sources='duke_masked',  # Use our custom name
        targets='duke_masked',
        height=256,
        width=128,
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        # Add random erasing - very helpful for Re-ID
        transforms=['random_flip', 'random_crop', 'random_erase'] 
    )

    print(f"Building Model: OSNet (AIN)...")

    # 4. Build Model
    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True 
    )

    model = model.cuda()

    # 5. Optimizer & Scheduler
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=25  # Adjusted for 60 epochs
    )

    # 6. Engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # 7. Run Training
    # We save to a DIFFERENT folder so we can compare later
    SAVE_DIR = 'log/osnet_duke_masked'
    
    print(f"Starting Training... Logs will be saved to {SAVE_DIR}")
    engine.run(
        save_dir=SAVE_DIR,
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

if __name__ == '__main__':
    main()