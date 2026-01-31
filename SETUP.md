# Setup Guide

## System Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- Minimum 2GB GPU memory recommended
- 4GB RAM or more

### Software
- **Operating System**: Linux, Windows 10/11, or macOS with NVIDIA GPU
- **CUDA Toolkit**: Version 11.0 or later
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **pip**: Latest version

## Installation Steps

### 1. Install CUDA Toolkit

#### Windows
1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the prompts
3. Verify installation:
```bash
nvcc --version
```

#### Linux
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Verify
nvcc --version
```

### 2. Set Up Python Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv gpu-env

# Activate
# Windows
gpu-env\Scripts\activate
# Linux/Mac
source gpu-env/bin/activate
```

#### Using conda
```bash
conda create -n gpu-env python=3.10
conda activate gpu-env
```

### 3. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install pycuda numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Verify PyCUDA Installation

```python
# Run in Python
import pycuda.autoinit
import pycuda.driver as cuda

print(f"CUDA Device: {cuda.Device(0).name()}")
print(f"Compute Capability: {cuda.Device(0).compute_capability()}")
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook GPU.ipynb
```

### Option 2: Google Colab
1. Upload `GPU.ipynb` to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU → Save

### Option 3: Kaggle
1. Create new notebook on Kaggle
2. Upload `GPU.ipynb`
3. Enable GPU: Settings → Accelerator: GPU

## Troubleshooting

### PyCUDA Installation Issues

**Error: `nvcc not found`**
```bash
# Add CUDA to PATH
# Windows (PowerShell)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"

# Linux/Mac (bash)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: `Python.h not found`**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Windows: Reinstall Python with "Include development headers" option
```

### Memory Issues

If you encounter GPU out-of-memory errors:
1. Reduce batch size in training loop
2. Use smaller models
3. Free GPU memory: `cuda.Context.pop()`

### CUDA Driver Issues

**Error: `CUDA driver version insufficient`**
- Update NVIDIA GPU drivers from [NVIDIA's website](https://www.nvidia.com/download/index.aspx)

## Performance Tips

1. **Verify GPU Usage**: Monitor with `nvidia-smi` command
2. **Optimize Memory**: Use smaller data types where possible (float32 vs float64)
3. **Batch Processing**: Increase batch size for better GPU utilization
4. **Kernel Tuning**: Adjust thread block sizes based on your GPU

## Additional Resources

- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Support

For issues specific to this project, please open an issue on the GitHub repository.
