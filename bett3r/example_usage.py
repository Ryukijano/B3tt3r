#!/usr/bin/env python3
"""
B3tt3r Example Usage Script
Demonstrates how to use B3tt3r for 3D reconstruction
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'mast3r'))
sys.path.append(str(Path(__file__).parent.parent / 'spann3r'))

from bett3r import Bett3R
from dust3r.utils.image import load_images
import matplotlib.pyplot as plt


def example_basic_usage():
    """Basic usage example with synthetic data"""
    print("=== B3tt3r Basic Usage Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize B3tt3r model
    print("Initializing B3tt3r model...")
    model = Bett3R.from_pretrained(
        'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
        use_feat=False,
        mem_pos_enc=False,
        memory_dropout=0.15,
        long_mem_size=1000,  # Smaller for demo
        work_mem_size=3
    ).to(device)
    
    model.eval()
    print("Model initialized successfully!")
    
    # Create synthetic image data
    print("Creating synthetic image pairs...")
    batch_size = 1
    height, width = 512, 512
    
    # Synthetic images (normally you'd load real images)
    img1 = torch.randn(batch_size, 3, height, width).to(device)
    img2 = torch.randn(batch_size, 3, height, width).to(device)
    
    view1 = {'img': img1}
    view2 = {'img': img2}
    
    # Process multiple image pairs to demonstrate memory accumulation
    print("Processing image pairs with spatial memory...")
    
    with torch.no_grad():
        for i in range(5):  # Process 5 pairs
            print(f"\nProcessing pair {i+1}/5...")
            
            # Add some variation to synthetic images
            view1['img'] = torch.randn(batch_size, 3, height, width).to(device)
            view2['img'] = torch.randn(batch_size, 3, height, width).to(device)
            
            # Forward pass with memory return
            pred1, pred2, memory = model(view1, view2, return_memory=True)
            
            # Print memory statistics
            if memory.mem_k is not None:
                print(f"  Memory size: {memory.mem_k.shape[1]}")
                print(f"  Working memory: {memory.wm}")
                print(f"  Long-term memory: {memory.lm}")
            else:
                print("  Memory not yet initialized")
            
            # Print prediction information
            print(f"  Prediction 1 keys: {list(pred1.keys())}")
            print(f"  Prediction 2 keys: {list(pred2.keys())}")
            
            if 'pts3d' in pred1:
                pts3d_1 = pred1['pts3d']
                pts3d_2 = pred2['pts3d']
                print(f"  Points 3D shape 1: {pts3d_1.shape}")
                print(f"  Points 3D shape 2: {pts3d_2.shape}")
    
    # Reset memory and show the difference
    print("\nResetting memory...")
    model.reset_memory()
    
    with torch.no_grad():
        pred1, pred2, memory = model(view1, view2, return_memory=True)
        if memory.mem_k is not None:
            print(f"Memory size after reset: {memory.mem_k.shape[1]}")
        else:
            print("Memory successfully reset")
    
    print("\n=== Basic Usage Example Completed ===")


def example_with_real_images():
    """Example with real images if available"""
    print("\n=== B3tt3r Real Images Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Look for sample images
    sample_paths = [
        Path('mast3r/dust3r/croco/assets/Chateau1.png'),
        Path('mast3r/dust3r/croco/assets/Chateau2.png')
    ]
    
    # Filter existing paths
    existing_paths = [p for p in sample_paths if p.exists()]
    
    if len(existing_paths) < 2:
        print("Real image example skipped: Need at least 2 sample images")
        return
    
    print(f"Found {len(existing_paths)} sample images")
    
    # Initialize model
    model = Bett3R.from_pretrained(
        'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    ).to(device)
    model.eval()
    
    # Load images
    print("Loading real images...")
    try:
        images = load_images([str(p) for p in existing_paths[:2]], size=512)
        
        view1 = {'img': torch.from_numpy(images[0]).unsqueeze(0).to(device)}
        view2 = {'img': torch.from_numpy(images[1]).unsqueeze(0).to(device)}
        
        print(f"Image shapes: {view1['img'].shape}, {view2['img'].shape}")
        
        # Forward pass
        with torch.no_grad():
            pred1, pred2, memory = model(view1, view2, return_memory=True)
        
        print("Successfully processed real images!")
        print(f"Prediction shapes - pts3d_1: {pred1.get('pts3d', 'N/A')}, pts3d_2: {pred2.get('pts3d', 'N/A')}")
        
        if memory.mem_k is not None:
            print(f"Memory accumulated: {memory.mem_k.shape[1]} features")
        
    except Exception as e:
        print(f"Error processing real images: {e}")
        print("This is normal if dependencies for image loading are missing")
    
    print("\n=== Real Images Example Completed ===")


def example_memory_comparison():
    """Compare B3tt3r with and without memory"""
    print("\n=== Memory Comparison Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize two models: one with memory, one without (by always resetting)
    print("Initializing models...")
    model_with_memory = Bett3R.from_pretrained(
        'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
        work_mem_size=3,
        long_mem_size=500
    ).to(device)
    
    model_without_memory = Bett3R.from_pretrained(
        'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
        work_mem_size=0,  # No working memory
        long_mem_size=0   # No long-term memory
    ).to(device)
    
    both_models = [model_with_memory, model_without_memory]
    model_names = ["With Memory", "Without Memory"]
    
    # Process the same sequence with both models
    print("Processing identical image sequences...")
    
    batch_size = 1
    height, width = 512, 512
    num_pairs = 3
    
    for model_idx, (model, name) in enumerate(zip(both_models, model_names)):
        print(f"\n--- Processing with {name} ---")
        model.eval()
        
        if name == "Without Memory":
            model.long_mem_size = 0
            model.work_mem_size = 0
        
        with torch.no_grad():
            for i in range(num_pairs):
                # Use the same random seed for reproducible "images"
                torch.manual_seed(42 + i)
                view1 = {'img': torch.randn(batch_size, 3, height, width).to(device)}
                torch.manual_seed(42 + i + 100)
                view2 = {'img': torch.randn(batch_size, 3, height, width).to(device)}
                
                pred1, pred2, memory = model(view1, view2, return_memory=True)
                
                if memory.mem_k is not None:
                    mem_size = memory.mem_k.shape[1]
                    print(f"  Pair {i+1}: Memory size = {mem_size}")
                else:
                    print(f"  Pair {i+1}: No memory stored")
                
                # Reset memory for "without memory" model
                if name == "Without Memory":
                    model.reset_memory()
    
    print("\n=== Memory Comparison Completed ===")


def main():
    """Run all examples"""
    print("B3tt3r Model Examples")
    print("====================")
    print("This script demonstrates different uses of the B3tt3r model")
    print("combining MASt3R stereo vision with SPANNer3R spatial memory.\n")
    
    try:
        # Basic usage with synthetic data
        example_basic_usage()
        
        # Real images if available
        example_with_real_images()
        
        # Memory comparison
        example_memory_comparison()
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        print("B3tt3r combines the best of MASt3R and SPANNer3R:")
        print("- MASt3R's powerful stereo vision capabilities")
        print("- SPANNer3R's sophisticated spatial memory")
        print("- No requirement for temporal image ordering")
        print("="*50)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This might be due to missing dependencies or model files.")
        print("Please ensure all required packages are installed.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()