# GPU-Accelerated Character Recognition with Custom CUDA Kernels

## Project Overview

High-performance character recognition system implementing Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) from scratch using custom CUDA kernels for GPU acceleration. Recognizes handwritten characters (N, A, R) from 5×5 binary pixel images.

## Key Features

- **Custom CUDA Kernels**: Hand-written CUDA C/C++ kernels for all neural network operations (forward propagation, backpropagation, convolution, pooling)
- **Dual Architecture Implementation**: Complete ANN and CNN implementations built from scratch
- **GPU Memory Management**: Efficient PyCUDA integration for optimal memory allocation and data transfer
- **Parallel Computing**: Leverages CUDA's parallel processing capabilities for accelerated training
- **Comprehensive Evaluation**: Performance metrics, confusion matrices, and visualization

## Model Performance

| Model | Accuracy | Training Time | Architecture |
|-------|----------|---------------|--------------|
| ANN   | 33.3%    | 1.18s        | 25→10→3      |
| CNN   | 44.4%    | 2.55s        | Conv32→Pool→FC10→FC3 |

*CNN achieves ~33% relative improvement over ANN*

## Architecture Details

### Artificial Neural Network (ANN)
- **Input Layer**: 25 neurons (5×5 flattened pixels)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 3 neurons with Softmax activation
- **Training**: Custom CUDA kernels for forward pass and backpropagation with atomic operations

### Convolutional Neural Network (CNN)
- **Input**: 5×5×1 grayscale images
- **Conv Layer**: 3×3 kernel, 32 filters, ReLU activation
- **Pooling Layer**: 2×2 max pooling
- **Fully Connected**: 32→10→3 with ReLU and Softmax
- **Training**: Complete CUDA implementation including gradient computation

## Technical Stack

- **GPU Computing**: PyCUDA 2024.1+
- **CUDA**: Custom C/C++ kernels for all neural operations
- **Data Processing**: NumPy 1.24.0+
- **Machine Learning**: scikit-learn 1.3.0+ (data splitting, metrics)
- **Visualization**: Matplotlib 3.7.0+, Seaborn 0.12.0+

## Requirements

```
pycuda>=2024.1
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Getting Started

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpu-character-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook GPU.ipynb
```

## Key Insights

- **Small Dataset Challenges**: With only 7 samples per class, neural networks struggle to learn generalizable patterns
- **CNN Advantage**: Even with minimal data, CNN's spatial feature extraction provides significant improvement
- **GPU Acceleration**: Custom CUDA kernels enable rapid experimentation and iteration
- **Parallel Processing**: Leveraging thousands of CUDA cores for matrix operations significantly speeds up training

## Implementation Highlights

### Custom CUDA Kernels

1. **ANN Forward Propagation**: Parallel matrix multiplication with ReLU/Softmax activation
2. **ANN Backpropagation**: Gradient computation with atomic operations for thread-safe updates
3. **CNN Convolution**: 2D convolution with boundary handling and parallel filter application
4. **CNN Max Pooling**: Sliding window max operation with gradient tracking
5. **Fully Connected Layers**: Optimized matrix operations for dense layers

### Memory Management

- Efficient GPU memory allocation using PyCUDA's `gpuarray`
- Minimized CPU-GPU data transfers
- Proper memory deallocation to prevent leaks

## Results & Visualization

The project includes comprehensive visualizations:
- **Confusion Matrices**: Model prediction patterns across classes
- **Performance Comparison**: Training time vs accuracy analysis
- **Sample Characters**: Visual representation of dataset variations

## Educational Value

This project demonstrates:
- Low-level GPU programming with CUDA
- Neural network implementation from first principles
- Parallel algorithm design and optimization
- Integration of compiled CUDA code with Python

## Future Enhancements

- Expand dataset for improved model generalization
- Implement data augmentation techniques
- Add batch normalization and dropout
- Optimize CUDA kernel performance with shared memory
- Extend to larger character sets

## Project Structure

```
gpu-character-recognition/
├── README.md              # Project documentation
├── GPU.ipynb              # Main notebook with CUDA implementation
├── requirements.txt       # Python dependencies
├── SETUP.md              # Detailed installation guide
└── .gitignore            # Git ignore rules
```
