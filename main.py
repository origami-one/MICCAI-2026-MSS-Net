import os
import argparse
import random
import time
import datetime
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Local imports
import options
import utils
from dataloader import create_dataloader
from utils.loss import compute_pos_weight_from_loader

# Fix GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    return total_params


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # 1. Configuration and Directory Setup
    parser = argparse.ArgumentParser(description='ECG Classification Training')
    opt = options.Options().init(parser).parse_args()
    set_seed(1234)

    run_id = f"{opt.ablation_mode}_{opt.env}"
    log_dir = os.path.join('log', f"{opt.arch}_{run_id}")
    model_dir = os.path.join(log_dir, 'models')
    utils.mkdir(model_dir)

    print(f"Config: {opt}")
    print(f"Start Time: {datetime.datetime.now().isoformat()}")

    # 2. Data Preparation
    train_loader, valid_loader = create_dataloader(opt)
    device = torch.device(opt.device)

    # 3. Model Initialization
    model = utils.get_arch(opt).to(device)
    count_parameters(model)
    trainer, validator = utils.get_train_mode(opt)

    # 4. Loss and Optimizer Setup
    # Calculate class distribution for Logit Adjustment
    cls_counts = torch.zeros(opt.class_num)
    for _, targets in train_loader:
        cls_counts += targets.sum(dim=0)
    cls_num_list = cls_counts.tolist()

    # Setup Loss with optional pos_weight for BCE
    pos_weight = None
    if opt.loss_type == 'base':
        pos_weight = compute_pos_weight_from_loader(train_loader, opt.class_num).to(device)

    loss_calculator = utils.get_loss(opt, weight=pos_weight, cls_num_list=cls_num_list).to(device)

    # Optimizer selection
    opt_name = opt.optimizer.lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_initial, weight_decay=opt.weight_decay)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr_initial, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_initial, weight_decay=opt.weight_decay)

    # Scheduler with Warmup
    warmup_epochs = 3
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, opt.nepoch - warmup_epochs), eta_min=1e-6)
    if opt.nepoch > warmup_epochs:
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
            main_scheduler
        ], milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    # Resume Training logic
    start_epoch = 1
    if opt.pretrained:
        chkpt = torch.load(opt.pretrain_model_path, map_location=device)
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        if 'scheduler' in chkpt: scheduler.load_state_dict(chkpt['scheduler'])
        start_epoch = chkpt.get('epoch', 0) + 1
        utils.optimizer_to(optimizer, device)

    # 5. Training Loop
    history = []
    best_metrics = {'auroc': 0.0, 'f1': 0.0, 'map': 0.0}
    writer = utils.Writer(os.path.join(log_dir, 'logs'))

    for epoch in range(start_epoch, opt.nepoch + 1):
        start_time = time.time()

        # Training Phase
        train_loss = trainer(model, train_loader, loss_calculator, optimizer, writer, epoch, device, opt)

        # Validation Phase with Memory Tracking
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

        val_res = validator(model, valid_loader, loss_calculator, writer, epoch, device, opt)

        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Extract Metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': val_res['loss'],
            'auc': val_res['auroc'],
            'map': val_res.get('auprc', 0.0),
            'f1': val_res['f1_opt'],
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(metrics)
        scheduler.step()

        # Logging
        duration = time.time() - start_time
        print(f"Epoch [{epoch}/{opt.nepoch}] {duration:.1f}s | Loss: {metrics['valid_loss']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | mAP: {metrics['map']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Peak Mem: {mem_peak:.1f}MB")

        # Save Checkpoints
        save_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            **metrics
        }
        torch.save(save_data, os.path.join(model_dir, 'last_checkpoint.pt'))

        for m_key in ['auroc', 'f1', 'map']:
            val_key = 'auc' if m_key == 'auroc' else m_key
            if metrics[val_key] > best_metrics[m_key]:
                best_metrics[m_key] = metrics[val_key]
                torch.save(save_data, os.path.join(model_dir, f'best_model_{m_key}.pt'))
                print(f"  New Best {m_key.upper()}: {best_metrics[m_key]:.4f}")

    # 6. Finalization: Save Results and Plot
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(log_dir, 'metrics.csv'), index=False)

    plt.figure(figsize=(18, 4))
    for i, col in enumerate(['train_loss', 'auc', 'map', 'f1'], 1):
        plt.subplot(1, 4, i)
        plt.plot(df['epoch'], df[col], marker='.')
        plt.title(col.replace('_', ' ').upper())
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'metrics_plot.png'), dpi=300)
    print(f"Training Complete. Logs saved to {log_dir}")