#!/usr/bin/env python3
"""
Evaluation script for B3tt3r model
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'mast3r'))
sys.path.append(str(Path(__file__).parent.parent / 'spann3r'))

from bett3r import Bett3R
from dust3r.datasets import get_data_loader
from dust3r.utils.image import load_images


def get_args_parser():
    parser = argparse.ArgumentParser('B3tt3r Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--checkpoint', required=True,
                        help='Path to B3tt3r checkpoint')
    parser.add_argument('--model_name', default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                        help='Base MASt3R model name')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--eval_dataset', default='BlendedMVS',
                        help='Evaluation dataset name')
    parser.add_argument('--eval_split', default='val',
                        help='Evaluation split')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./results/bett3r_eval',
                        help='Output directory for results')
    parser.add_argument('--save_pointclouds', action='store_true',
                        help='Save generated point clouds')
    
    return parser


def evaluate_model(model, data_loader, device, save_pointclouds=False, output_dir=None):
    """Evaluate B3tt3r model on dataset"""
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Extract image pairs
            view1 = {'img': batch['view1']['img']}
            view2 = {'img': batch['view2']['img']}
            
            if 'true_shape' in batch['view1']:
                view1['true_shape'] = batch['view1']['true_shape']
                view2['true_shape'] = batch['view2']['true_shape']
            
            # Forward pass
            pred1, pred2, memory = model(view1, view2, return_memory=True)
            
            # Log memory statistics
            if memory.mem_k is not None:
                print(f"Batch {batch_idx}: Memory size = {memory.mem_k.shape[1]}, "
                      f"Working memory = {memory.wm}, Long-term memory = {memory.lm}")
            
            # Save point clouds if requested
            if save_pointclouds and output_dir:
                output_path = Path(output_dir) / f'pointcloud_{batch_idx:04d}.ply'
                save_pointcloud(pred1, pred2, output_path)
            
            num_samples += 1
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} batches")
    
    print(f"Evaluation completed. Processed {num_samples} samples.")
    return total_loss / num_samples if num_samples > 0 else 0.0


def save_pointcloud(pred1, pred2, output_path):
    """Save predictions as point cloud"""
    # Extract 3D points and colors
    pts3d_1 = pred1.get('pts3d', None)
    pts3d_2 = pred2.get('pts3d', None)
    
    if pts3d_1 is not None and pts3d_2 is not None:
        # Combine points from both views
        points = torch.cat([pts3d_1.flatten(0, 1), pts3d_2.flatten(0, 1)], dim=0)
        points = points.cpu().numpy()
        
        # Simple PLY format export
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")


def demo_inference(model, image_paths, device):
    """Demo inference on image pair"""
    print(f"Running demo inference on {len(image_paths)} images...")
    
    # Load images
    images = load_images(image_paths, size=512)
    
    if len(images) < 2:
        print("Need at least 2 images for demo")
        return
    
    # Prepare views
    view1 = {'img': torch.from_numpy(images[0]).unsqueeze(0).to(device)}
    view2 = {'img': torch.from_numpy(images[1]).unsqueeze(0).to(device)}
    
    with torch.no_grad():
        pred1, pred2, memory = model(view1, view2, return_memory=True)
    
    print("Demo inference completed!")
    print(f"Prediction 1 keys: {pred1.keys()}")
    print(f"Prediction 2 keys: {pred2.keys()}")
    
    if memory.mem_k is not None:
        print(f"Memory size: {memory.mem_k.shape[1]}")
    
    return pred1, pred2, memory


def main(args):
    print(f"Evaluating B3tt3r model:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.eval_dataset}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.save_pointclouds:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Initialize model
    print("Initializing model...")
    if 'args' in checkpoint:
        # Use saved model configuration
        saved_args = checkpoint['args']
        model = Bett3R.from_pretrained(
            args.model_name,
            use_feat=getattr(saved_args, 'use_feat', False),
            mem_pos_enc=getattr(saved_args, 'mem_pos_enc', False),
            memory_dropout=getattr(saved_args, 'memory_dropout', 0.15),
            long_mem_size=getattr(saved_args, 'long_mem_size', 4000),
            work_mem_size=getattr(saved_args, 'work_mem_size', 5)
        ).to(device)
    else:
        # Use default configuration
        model = Bett3R.from_pretrained(args.model_name).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Initialize data loader
    print("Loading evaluation data...")
    try:
        eval_loader = get_data_loader(
            dataset=args.eval_dataset,
            split=args.eval_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Run evaluation
        avg_loss = evaluate_model(model, eval_loader, device, 
                                 args.save_pointclouds, args.output_dir)
        print(f"Average evaluation loss: {avg_loss:.4f}")
        
    except Exception as e:
        print(f"Could not load evaluation dataset: {e}")
        print("Running demo inference instead...")
        
        # Demo with sample images if available
        sample_images = [
            'mast3r/dust3r/croco/assets/Chateau1.png',
            'mast3r/dust3r/croco/assets/Chateau2.png'
        ]
        
        # Check if sample images exist
        sample_images = [img for img in sample_images if Path(img).exists()]
        
        if sample_images:
            demo_inference(model, sample_images, device)
        else:
            print("No sample images found for demo")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)