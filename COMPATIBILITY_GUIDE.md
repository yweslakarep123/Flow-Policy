# Panduan Kompatibilitas FlowPolicy

## Perubahan yang Dilakukan

### 1. Struktur Kode
- **FlowPolicy Model**: Menggunakan implementasi original dari `policy/flowpolicy.py`
- **Dataset Loader**: Ditambahkan `utils/dataset_loader.py` untuk load RoboSet_Sim .h5 files
- **Import Paths**: Diperbaiki agar kompatibel dengan struktur folder yang ada

### 2. Dataset
- Dataset RoboSet_Sim disimpan di `dataset/RoboSet_Sim/`
- Format: `.h5` files dengan keys: `obs`, `action`, dll
- Support untuk task-specific loading

### 3. Training Script
- `train.py` telah diupdate untuk:
  - Load dataset dari RoboSet_Sim
  - Menggunakan FlowPolicy original
  - Kompatibel dengan shape_meta format
  - Normalizer setup otomatis

## Cara Menggunakan

### 1. Training dengan RoboSet Dataset

```bash
python train.py \
    --data_path dataset/RoboSet_Sim \
    --task_name FK1_MicroOpenRandom_v2d-v4 \
    --horizon 16 \
    --n_action_steps 8 \
    --n_obs_steps 1 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 100 \
    --use_pc_color \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

### 2. Training dengan Multiple Tasks

```bash
# Load dari folder human_demos_by_task
python train.py \
    --data_path dataset/RoboSet_Sim/FK1-v4\(human\)/human_demos_by_task \
    --task_name FK1_MicroOpenRandom_v2d-v4 \
    --horizon 16 \
    --batch_size 32 \
    --epochs 100
```

### 3. Parameter Penting

#### Dataset Parameters
- `--data_path`: Path ke folder dataset atau file .h5
- `--task_name`: Nama task (optional, untuk filter task tertentu)
- `--horizon`: Horizon untuk action prediction (default: 16)

#### Model Parameters
- `--encoder_output_dim`: Output dimension encoder (default: 256)
- `--use_pc_color`: Gunakan RGB color dalam point cloud
- `--n_action_steps`: Number of action steps (default: 8)
- `--n_obs_steps`: Number of observation steps (default: 1)

#### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 100)
- `--weight_decay`: Weight decay (default: 1e-5)

#### Consistency Flow Matching Parameters
- `--eps`: Epsilon untuk CFM (default: 0.01)
- `--num_segments`: Number of segments (default: 2)
- `--num_inference_step`: Number of inference steps (default: 1)

## Struktur Dataset yang Diharapkan

Dataset .h5 files harus memiliki keys berikut:

### Required Keys:
- `action`: Actions array, shape (T, action_dim)
- `obs`: Observations dict atau array

### Observation Format:
Jika `obs` adalah dict, harus mengandung:
- `point_cloud`: Point cloud data, shape (T, N, 3) atau (T, N, 6) jika dengan RGB
- `agent_pos`: Robot state, shape (T, state_dim)

### Contoh:
```python
# .h5 file structure
{
    'action': np.array,  # (T, action_dim)
    'obs': {
        'point_cloud': np.array,  # (T, N, 3) or (T, N, 6)
        'agent_pos': np.array,    # (T, state_dim)
        # ... other obs keys
    }
}
```

## Troubleshooting

### Error: "No valid data files found"
- Pastikan path ke dataset benar
- Cek apakah file .h5 ada di folder tersebut
- Pastikan format .h5 file sesuai dengan yang diharapkan

### Error: "Key 'point_cloud' not found"
- Pastikan observations memiliki key 'point_cloud'
- Atau adjust shape_meta manual di train.py

### Error: Import errors
- Pastikan semua dependencies terinstall
- Cek Python path sudah include root directory
- Pastikan semua file ada di folder yang benar

## Catatan

1. **Point Cloud Format**: 
   - Default: (N, 3) untuk XYZ coordinates
   - Dengan RGB: (N, 6) untuk XYZRGB
   - Gunakan `--use_pc_color` untuk enable RGB

2. **Normalizer**:
   - Normalizer akan di-fit otomatis dari sample dataset
   - Pastikan dataset cukup besar (minimal 100 samples untuk normalizer)

3. **Memory**:
   - Dataset akan di-load ke memory
   - Untuk dataset besar, pertimbangkan menggunakan ReplayBuffer dengan disk caching

## Next Steps

- [ ] Implement inference script yang kompatibel
- [ ] Update environment wrapper untuk RoboHive
- [ ] Add evaluation metrics
- [ ] Add visualization utilities
