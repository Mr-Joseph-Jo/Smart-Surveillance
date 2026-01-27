import torchreid
import sys
import os.path as osp
from torchreid.data import ImageDataset

# 1. Define a Custom Dataset Class with RELABELING
class DukePatches(ImageDataset):
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        
        train_dir = osp.join(self.root, 'bounding_box_train')
        query_dir = osp.join(self.root, 'query')
        gallery_dir = osp.join(self.root, 'bounding_box_test')
        
        if not osp.exists(train_dir):
            raise RuntimeError(f"'{train_dir}' does not exist.")

        # 1. Load raw data
        train_raw = self.process_dir(train_dir)
        query = self.process_dir(query_dir)
        gallery = self.process_dir(gallery_dir)

        # 2. RELABEL TRAINING DATA (The Fix)
        # Convert IDs like [1, 5, 10] -> [0, 1, 2]
        train = self.relabel_dataset(train_raw)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        import glob
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []
        for img_path in img_paths:
            filename = osp.basename(img_path)
            parts = filename.split('_')
            pid = int(parts[0])
            camid = int(parts[1].replace('c', ''))
            data.append((img_path, pid, camid))
        return data

    def relabel_dataset(self, dataset):
        # Get all unique PIDs
        pids = sorted(list(set([pid for _, pid, _ in dataset])))
        
        # Create mapping: Real_ID -> 0..N index
        pid2label = {pid: i for i, pid in enumerate(pids)}
        
        new_dataset = []
        for img_path, pid, camid in dataset:
            # Replace Real_ID with the new continuous label
            new_dataset.append((img_path, pid2label[pid], camid))
            
        return new_dataset

# 2. Register the custom dataset name
torchreid.data.register_image_dataset('duke_patches', DukePatches)

def train(target_type):
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING FOR: {target_type.upper()}")
    print(f"{'='*60}")

    PATCH_ROOT = f"./duke_patches/{target_type}"
    SAVE_DIR = f"log/osnet_duke_{target_type}"

    datamanager = torchreid.data.ImageDataManager(
        root=PATCH_ROOT,        
        sources='duke_patches', 
        targets='duke_patches',
        height=128,             
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip']
    )

    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    ).cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='single_step')
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    print(f"Saving logs to: {SAVE_DIR}")
    engine.run(
        save_dir=SAVE_DIR,
        max_epoch=40,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

if __name__ == '__main__':
    #train('hair')
    train('face')