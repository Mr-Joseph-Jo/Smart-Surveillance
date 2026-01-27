import torchreid
import torch

def main():
    # 1. Configuration
    DATASET_PATH = "./duke" 
    MAX_EPOCHS = 50           
    BATCH_SIZE = 32           
    
    print(f"Initializing Data Manager for DukeMTMC-reID...")

    # 2. Data Manager
    # Note: On Windows, num_workers=0 is the safest option if you still have issues,
    # but wrapping in main() usually allows num_workers > 0.
    datamanager = torchreid.data.ImageDataManager(
        root=DATASET_PATH,
        sources='dukemtmcreid',
        targets='dukemtmcreid',
        height=256,
        width=128,
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )

    print(f"Building Model: OSNet (AIN)...")

    # 3. Build Model
    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True 
    )

    # Move to GPU
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
        stepsize=20
    )

    # 5. Engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # 6. Run Training
    print("Starting Training (This may take 1-2 hours depending on GPU)...")
    engine.run(
        save_dir='log/osnet_duke',
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

# --- VITAL FOR WINDOWS ---
if __name__ == '__main__':
    main()