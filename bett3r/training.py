#!/usr/bin/env python3
"""
Training script for B3tt3r model
Combines MASt3R stereo vision with SPANNer3R spatial memory
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'mast3r'))
sys.path.append(str(Path(__file__).parent.parent / 'spann3r'))

from bett3r import Bett3R
from dust3r.datasets import get_data_loader
from spann3r.spann3r.loss import Regr3D_t_ScaleShiftInv
from mast3r.losses import *


def get_args_parser():
    parser = argparse.ArgumentParser('B3tt3r Training', add_help=False)
    
    # Model parameters
    parser.add_argument('--model_name', default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                        help='Pretrained MASt3R model to use as backbone')
    parser.add_argument('--use_feat', action='store_true', default=False,
                        help='Use feature-based memory encoding')
    parser.add_argument('--mem_pos_enc', action='store_true', default=False,
                        help='Use positional encoding in memory')
    parser.add_argument('--memory_dropout', type=float, default=0.15,
                        help='Dropout rate for memory operations')
    parser.add_argument('--long_mem_size', type=int, default=4000,
                        help='Long-term memory size')
    parser.add_argument('--work_mem_size', type=int, default=5,
                        help='Working memory size')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    
    # Data parameters
    parser.add_argument('--train_dataset', default='BlendedMVS',
                        help='Training dataset name')
    parser.add_argument('--train_split', default='train',
                        help='Training split')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Paths
    parser.add_argument('--output_dir', default='./checkpoints/bett3r',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    
    return parser


def main(args):
    print(f"Training B3tt3r model with config:")
    print(f"  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Memory config: long={args.long_mem_size}, work={args.work_mem_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("Initializing B3tt3r model...")
    model = Bett3R.from_pretrained(
        args.model_name,
        use_feat=args.use_feat,
        mem_pos_enc=args.mem_pos_enc,
        memory_dropout=args.memory_dropout,
        long_mem_size=args.long_mem_size,
        work_mem_size=args.work_mem_size
    ).to(device)
    
    # Initialize loss function
    criterion = Regr3D_t_ScaleShiftInv()
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize data loader
    print("Loading training data...")
    train_loader = get_data_loader(
        dataset=args.train_dataset,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Extract image pairs from batch
            view1 = {'img': batch['view1']['img']}
            view2 = {'img': batch['view2']['img']}
            
            if 'true_shape' in batch['view1']:
                view1['true_shape'] = batch['view1']['true_shape']
                view2['true_shape'] = batch['view2']['true_shape']
            
            # Forward pass
            optimizer.zero_grad()
            pred1, pred2 = model(view1, view2)
            
            # Compute loss
            loss = criterion(batch, [pred1, pred2])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / 'bett3r_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)