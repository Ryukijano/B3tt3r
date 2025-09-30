#!/bin/bash
# Script to run tensor shape validation tests for B3tt3r

echo "=================================="
echo "B3tt3r Tensor Shape Tests"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

echo "Checking dependencies..."

# Check if PyTorch is installed
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found. Installing dependencies..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy==1.26.4 tqdm scipy roma einops
else
    echo "✓ PyTorch is installed"
fi

echo ""
echo "Running shape tests..."
echo ""

# Run the tests
python3 test_shapes.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ All tests passed successfully!"
else
    echo ""
    echo "✗ Some tests failed. Check the output above."
fi

exit $exit_code
