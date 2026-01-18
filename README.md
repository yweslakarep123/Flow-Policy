# FlowPolicy: 3D Flow-based Policy for Franka Kitchen

Implementasi FlowPolicy untuk environment Franka Kitchen berbasis vision menggunakan Consistency Flow Matching berdasarkan paper [FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation](2412.04987v2.pdf).

## Fitur

- **3D Vision Representation**: Menggunakan Point Cloud Encoder (berbasis PointNet++) untuk representasi visual 3D
- **Consistency Flow Matching**: Single-step inference untuk efisiensi tinggi (7x lebih cepat dibanding diffusion-based methods)
- **Velocity Field Normalization**: Normalisasi self-consistency pada velocity field untuk straight-line flows
- **Franka Kitchen Support**: Wrapper untuk environment Franka Kitchen dengan vision support

## Instalasi

### Menggunakan Conda (Disarankan untuk CUDA)

```bash
# Buat environment dari environment.yml
conda env create -f environment.yml
conda activate flowpolicy

# Verifikasi CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Menggunakan pip (Alternatif)

```bash
# Install PyTorch dengan CUDA terlebih dahulu (pilih sesuai CUDA version)
# Untuk CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies lainnya
pip install -r requirements.txt
```

**Catatan**: Untuk dukungan CUDA terbaik, gunakan Conda. Lihat [SETUP.md](SETUP.md) untuk panduan lengkap.

**Versi Python**: Python 3.10 atau 3.11 (disarankan 3.10.9)

## Penggunaan

### Training

```bash
# Basic training
python train.py --env_name FrankaKitchen-v1 --num_demos 50 --epochs 100

# Dengan RGB features
python train.py --env_name FrankaKitchen-v1 --num_demos 50 --epochs 100 --use_rgb

# Custom configuration
python train.py \
    --env_name FrankaKitchen-v1 \
    --tasks microwave kettle \
    --num_demos 50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_points 1024 \
    --use_rgb
```

### Inference

```bash
# Basic inference
python inference.py --checkpoint_path checkpoints/best_model.pt

# Dengan rendering
python inference.py --checkpoint_path checkpoints/best_model.pt --render

# Multiple episodes
python inference.py --checkpoint_path checkpoints/best_model.pt --num_episodes 20
```

## Struktur Proyek

```
FlowPolicy/
├── models/
│   ├── point_cloud_encoder.py   # Point Cloud Encoder (PointNet++)
│   ├── flow_matching.py         # Consistency Flow Matching model
│   └── __init__.py
├── policies/
│   ├── flow_policy.py           # FlowPolicy main implementation
│   └── __init__.py
├── envs/
│   ├── franka_kitchen_wrapper.py  # Franka Kitchen wrapper dengan vision
│   └── __init__.py
├── utils/
│   ├── data_utils.py            # Data processing utilities
│   ├── training_utils.py        # Training utilities
│   └── __init__.py
├── train.py                     # Training script
├── inference.py                 # Inference script
├── config.py                    # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Arsitektur

### FlowPolicy Components

1. **Point Cloud Encoder**: Encodes 3D point cloud observations ke feature vectors
2. **Consistency Flow Matching**: Models straight-line flows dari noise ke action space
3. **Velocity Network**: Neural network untuk memprediksi velocity field

### Key Features

- **Single-step inference**: Menggunakan consistency flow matching untuk generate actions dalam 1 step
- **3D vision conditioning**: Conditions pada 3D point cloud observations
- **Velocity normalization**: Normalisasi self-consistency untuk straight-line flows

## Catatan Implementasi

- Environment wrapper mengkonversi RGB-D observations ke point clouds
- Point cloud encoder menggunakan architecture berbasis PointNet++
- Flow matching menggunakan straight-line flows dengan velocity consistency
- Training menggunakan flow matching loss dengan timestep sampling

## Referensi

Paper: [FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation](https://arxiv.org/abs/2412.04987)
