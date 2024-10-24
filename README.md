# B3tt3r
A 3-D reconstruction paradigm combining Mast3r and Spann3r models to make it B3tt3r.

## Combining Mast3r and Spann3r Models for 3D Reconstruction

This repository demonstrates how to combine the Mast3r and Spann3r models for 3D reconstruction. The Mast3r model is used for feature extraction and initial 3D point cloud generation, while the Spann3r model is used for refining the 3D reconstruction using spatial memory.

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- Open3D
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ryukijano/B3tt3r.git
   cd B3tt3r
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training

To train the combined Mast3r and Spann3r models, run the following command:
```bash
python spann3r/training.py --config configs/train_config.yaml
```

#### Evaluation

To evaluate the combined model, run the following command:
```bash
python spann3r/evaluate.py --config configs/eval_config.yaml
```

### Example

Here is an example of how to use the combined model for 3D reconstruction:

```python
import torch
from spann3r.model import Spann3R
from spann3r.datasets import load_dataset

# Load the dataset
dataset = load_dataset('path/to/dataset')

# Initialize the model
model = Spann3R()

# Set the model to evaluation mode
model.eval()

# Perform 3D reconstruction
with torch.no_grad():
    for frames in dataset:
        preds, preds_all = model(frames)
        # Process the predictions
        # ...
```

### Expected Outputs

The combined model should produce a refined 3D point cloud with improved accuracy and completeness compared to using the Mast3r model alone. The output point cloud can be visualized using Open3D or other 3D visualization tools.

### Acknowledgements

This work is based on the Mast3r and Spann3r models. We would like to thank the authors of these models for their contributions to the field of 3D reconstruction.
