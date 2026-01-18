# Setup Guide: FlowPolicy dengan Conda dan CUDA

Panduan lengkap untuk setup environment FlowPolicy menggunakan Conda dengan dukungan CUDA.

## Prerequisites

- NVIDIA GPU dengan CUDA support
- Conda atau Miniconda terinstall
- CUDA Toolkit 11.8 atau 12.1 (tergantung GPU)

## Versi Python yang Disarankan

**Python 3.10.9** (disarankan) atau **Python 3.11** untuk kompatibilitas terbaik dengan PyTorch CUDA dan Gymnasium-Robotics.

## Metode Setup

### Metode 1: Menggunakan environment.yml (Disarankan)

Ini adalah metode termudah dan paling direkomendasikan:

```bash
# Clone repository (jika belum)
cd /path/to/FlowPolicy

# Buat conda environment dari file environment.yml
conda env create -f environment.yml

# Aktivasi environment
conda activate flowpolicy

# Verifikasi CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### Metode 2: Manual Setup dengan Conda

Jika ingin lebih kontrol atau menggunakan CUDA version berbeda:

```bash
# Buat environment dengan Python 3.10
conda create -n flowpolicy python=3.10.9 -y

# Aktivasi environment
conda activate flowpolicy

# Install PyTorch dengan CUDA 11.8 (atau ganti dengan 12.1 jika diperlukan)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies lainnya via pip
pip install -r requirements.txt

# Verifikasi CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Metode 3: Menggunakan pip (Alternatif)

Jika conda tidak tersedia, bisa menggunakan pip tapi kurang disarankan untuk CUDA:

```bash
# Buat virtual environment dengan Python 3.10
python3.10 -m venv flowpolicy_env
source flowpolicy_env/bin/activate  # Linux/Mac
# atau
flowpolicy_env\Scripts\activate  # Windows

# Install PyTorch dengan CUDA (pilih sesuai CUDA version GPU Anda)
# Untuk CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Untuk CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies lainnya
pip install -r requirements.txt

# Verifikasi CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verifikasi Setup

Setelah instalasi, verifikasi setup dengan script berikut:

```bash
conda activate flowpolicy
python -c "
import torch
import numpy as np
import gymnasium

print('=' * 50)
print('Verifikasi Environment FlowPolicy')
print('=' * 50)
print(f'Python version: {__import__(\"sys\").version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'NumPy version: {np.__version__}')
print(f'Gymnasium version: {gymnasium.__version__}')
print('=' * 50)
"
```

Expected output jika berhasil:
```
==================================================
Verifikasi Environment FlowPolicy
==================================================
Python version: 3.10.9
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3090
GPU memory: 24.00 GB
NumPy version: 1.24.3
Gymnasium version: 0.29.1
==================================================
```

## Troubleshooting

### CUDA tidak terdeteksi

1. **Cek CUDA driver:**
   ```bash
   nvidia-smi
   ```

2. **Cek CUDA version yang terinstall:**
   ```bash
   nvcc --version
   ```

3. **Pastikan PyTorch CUDA version cocok dengan CUDA driver:**
   - CUDA 11.8 driver → gunakan `pytorch-cuda=11.8`
   - CUDA 12.1+ driver → gunakan `pytorch-cuda=12.1`

### Error saat install MuJoCo

MuJoCo memerlukan MuJoCo engine terpisah:

```bash
# Install MuJoCo Python bindings
pip install mujoco

# Verifikasi
python -c "import mujoco; print('MuJoCo installed successfully')"
```

### Error saat import gymnasium-robotics

Pastikan semua dependencies terinstall:

```bash
pip install --upgrade gymnasium gymnasium-robotics
```

### Update Environment

Untuk update environment setelah perubahan `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Catatan Penting

1. **CUDA Version**: Pastikan PyTorch CUDA version cocok dengan CUDA driver GPU Anda
2. **Python Version**: Gunakan Python 3.10 atau 3.11 untuk kompatibilitas terbaik
3. **Conda vs Pip**: Untuk CUDA support, Conda lebih disarankan karena bisa mengelola CUDA toolkit
4. **Memory**: Pastikan GPU memiliki cukup memory (minimal 4GB untuk training)

## Next Steps

Setelah environment siap, lanjutkan ke training:

```bash
conda activate flowpolicy
python train.py --env_name FrankaKitchen-v1 --num_demos 50 --epochs 100 --use_rgb
```
