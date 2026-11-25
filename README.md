# SAIST

Training and inference use `main.py`, supporting NUAA-SIRST, NUDT-SIRST, and IRSTD-1k. Run the following commands from the project root directory.

## Environment Setup

Linux and Python 3.10.

```bash
conda create -n saist310 python=3.10 -y
conda activate saist310
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Downloads

[Download the MIRSTD dataset](https://huggingface.co/datasets/steve1325467/MIRSTD/resolve/main/MIRSTD.zip?download=true)

| Dataset | Model Weights |
|---|---|
| NUAA-SIRST | [NUAA_best_val.pt](https://huggingface.co/steve1325467/SAIST/resolve/main/NUAA_best_val.pt?download=true) |
| NUDT-SIRST | [NUDT_best_val.pt](https://huggingface.co/steve1325467/SAIST/resolve/main/NUDT_best_val.pt?download=true) |
| IRSTD-1k | [IRSTD1k_best_val.pt](https://huggingface.co/steve1325467/SAIST/resolve/main/IRSTD1k_best_val.pt?download=true) |

Place the three downloaded checkpoints in the project's `model_pt` directory.

## Path Configuration

Place the pretrained SAM ViT-L and CLIP checkpoints at the following locations. Training requires both files; inference with a full trained checkpoint still requires the CLIP file.

```text
pretrained_checkpoint/sam_vit_l_0b3195.pth
pretrained_checkpoint/ViT-B-32.pt
```

Set the actual paths to the dataset and split directories:

```bash
export DATA_ROOT=/path/to/MIRSTD
export SPLIT_ROOT=/path/to/DataSplit
```

Each dataset directory contains `images`, `masks`, and `descriptions`. Each corresponding split directory contains `train.txt`, `val.txt`, and `test.txt`, with one image ID per line.

```text
MIRSTD/
├── NUAA-SIRST/
├── NUDT-SIRST/
└── IRSTD-1k/

DataSplit/
├── NUAA/
├── NUDT/
└── IRSTD-1k/
```
## Inference

The following commands use the checkpoints downloaded from Hugging Face to evaluate the corresponding test splits.

**NUAA-SIRST**

```bash
python -u main.py --mode infer --dataset nuaa --weights model_pt/NUAA_best_val.pt --data-root "$DATA_ROOT/NUAA-SIRST" --split-dir "$SPLIT_ROOT/NUAA" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```

**NUDT-SIRST**

```bash
python -u main.py --mode infer --dataset nudt --weights model_pt/NUDT_best_val.pt --data-root "$DATA_ROOT/NUDT-SIRST" --split-dir "$SPLIT_ROOT/NUDT" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```

**IRSTD-1k**

```bash
python -u main.py --mode infer --dataset irstd1k --weights model_pt/IRSTD1k_best_val.pt --data-root "$DATA_ROOT/IRSTD-1k" --split-dir "$SPLIT_ROOT/IRSTD-1k" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```

The reported metrics are IoU, Pd, and Fa (×10⁻⁶).
