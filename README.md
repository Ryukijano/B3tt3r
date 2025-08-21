# B3tt3r
A 3-D reconstruction paradigm combining MASt3R and SPANNer3R models to make it B3tt3r.

## Combining MASt3R and SPANNer3R Models for 3D Reconstruction

This repository demonstrates how to combine the MASt3R and SPANNer3R models for enhanced 3D reconstruction. The MASt3R model provides powerful stereo vision capabilities for feature extraction and initial 3D point cloud generation, while the SPANNer3R model contributes sophisticated spatial memory mechanisms for maintaining and refining 3D understanding across multiple views.

## Introducing Bett3r

Bett3r is an advanced extension of the MASt3R model that incorporates sophisticated spatial memory features inspired by SPANNer3R. Unlike SPANNer3R, Bett3r does not confine images to a certain temporal order or require strict photogrammetry sequences. This flexibility allows for more versatile and efficient 3D reconstruction from arbitrary image pairs.

### Key Features

- **Sophisticated Spatial Memory**: Implements working memory and long-term memory with intelligent pruning
- **Similarity Checking**: Avoids storing redundant features using cosine similarity matching
- **Memory Attention**: Advanced attention mechanisms for memory read/write operations
- **Flexible Processing**: No requirement for temporal image ordering like SPANNer3R
- **MASt3R Integration**: Leverages MASt3R's powerful stereo vision backbone
- **Memory Pruning**: Intelligent memory management based on attention weights and usage statistics

### Architecture Overview

Bett3r combines:
1. **MASt3R Backbone**: For robust stereo feature extraction and 3D point estimation
2. **Advanced Spatial Memory**: Multi-level memory system with working and long-term components
3. **Feature Encoding**: Sophisticated key-value encoding for memory operations
4. **Attention-based Retrieval**: Memory querying using learned attention mechanisms

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ryukijano/B3tt3r.git
   cd B3tt3r
   ```

2. Initialize submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Install the required dependencies:
   ```bash
   pip install -r spann3r/requirements.txt
   pip install -r mast3r/requirements.txt
   ```

### Usage

#### Basic Usage

```python
import torch
from bett3r.bett3r.model import Bett3R

# Initialize the model
model = Bett3R.from_pretrained(
    'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
    use_feat=False,
    mem_pos_enc=False,
    memory_dropout=0.15,
    long_mem_size=4000,
    work_mem_size=5
)

# Prepare image views
view1 = {'img': torch.randn(1, 3, 512, 512)}  # Replace with real images
view2 = {'img': torch.randn(1, 3, 512, 512)}

# Process with spatial memory
model.eval()
with torch.no_grad():
    pred1, pred2, memory = model(view1, view2, return_memory=True)

# Access 3D predictions
pts3d_1 = pred1['pts3d']  # 3D points for view 1
pts3d_2 = pred2['pts3d']  # 3D points for view 2

print(f"Memory accumulated: {memory.mem_k.shape[1] if memory.mem_k else 0} features")
```

#### Training

To train the Bett3r model, run the following command:
```bash
python bett3r/training.py --config configs/train_config.yaml
```

#### Evaluation

To evaluate the model, run:
```bash
python bett3r/evaluate.py --checkpoint path/to/checkpoint.pth
```

#### Testing

To test the implementation:
```bash
python bett3r/test_bett3r.py
```

### Example

Here's a comprehensive example showing the key features:

```python
import torch
from bett3r.bett3r.model import Bett3R, SpatialMemory

# Load the dataset
# dataset = load_dataset('path/to/dataset')  # Your dataset loading

# Initialize the model with memory configuration
model = Bett3R.from_pretrained(
    'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
    long_mem_size=2000,  # Long-term memory size
    work_mem_size=5,     # Working memory size
    memory_dropout=0.1   # Dropout for memory operations
)

# Set the model to evaluation mode
model.eval()

# Process multiple image pairs to build spatial memory
with torch.no_grad():
    for i, (view1, view2) in enumerate(image_pairs):
        pred1, pred2, memory = model(view1, view2, return_memory=True)
        
        print(f"Pair {i}: Memory size = {memory.mem_k.shape[1] if memory.mem_k else 0}")
        print(f"Working memory: {memory.wm}, Long-term memory: {memory.lm}")
        
        # Process predictions
        pts3d_1 = pred1['pts3d']
        pts3d_2 = pred2['pts3d']
        
        # Your downstream processing...

# Reset memory for new scene
model.reset_memory()
```

### Memory Management

Bett3r implements sophisticated memory management:

- **Working Memory**: Stores recent features for immediate access
- **Long-term Memory**: Maintains important features based on attention weights
- **Similarity Checking**: Prevents redundant feature storage
- **Automatic Pruning**: Removes less important memories when capacity is reached

### Expected Outputs

The Bett3r model produces:
1. **Enhanced 3D Point Clouds**: Improved accuracy through spatial memory
2. **Confidence Maps**: Per-pixel confidence scores
3. **Memory Statistics**: Information about memory usage and efficiency
4. **Feature Enhancements**: Memory-enhanced feature representations

Compared to using MASt3R alone, Bett3r provides:
- Better temporal consistency across multiple views
- Reduced redundancy in feature processing
- Enhanced robustness through memory mechanisms
- More accurate 3D reconstructions

### Advanced Configuration

```python
# Advanced memory configuration
model = Bett3R.from_pretrained(
    model_name,
    use_feat=True,           # Use feature-based memory encoding
    mem_pos_enc=True,        # Enable positional encoding in memory
    memory_dropout=0.15,     # Memory dropout rate
    long_mem_size=4000,      # Maximum long-term memory size
    work_mem_size=5,         # Working memory capacity
)

# Memory can be manually managed
model.reset_memory()  # Clear all memory
```

### Performance Notes

- **Memory Efficiency**: Automatic pruning keeps memory usage bounded
- **Similarity Thresholding**: Configurable similarity threshold prevents redundant storage
- **Attention-based Retrieval**: Efficient memory access using learned attention
- **GPU Acceleration**: Full CUDA support for memory operations

### Acknowledgements

This work builds upon and combines innovations from:
- **MASt3R**: For robust stereo vision and 3D reconstruction capabilities
- **SPANNer3R**: For sophisticated spatial memory mechanisms and temporal consistency
- **DUSt3R**: As the foundational architecture for both MASt3R and SPANNer3R

We thank the authors of these models for their contributions to 3D computer vision and their open-source implementations that made this work possible.
