# SAIST

训练与推理均使用 `main.py`，支持 NUAA-SIRST、NUDT-SIRST 和 IRSTD-1k。以下命令在项目根目录执行。

## 安装环境

Linux、Python 3.10，使用支持 BF16 的 NVIDIA GPU。

```bash
conda create -n saist310 python=3.10 -y
conda activate saist310
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 配置路径

将 SAM ViT-L 和 CLIP 的预训练文件放在以下位置：

```text
pretrained_checkpoint/sam_vit_l_0b3195.pth
pretrained_checkpoint/ViT-B-32.pt
```

设置实际路径：

```bash
export DATA_ROOT=/path/to/MIRSTD
export SPLIT_ROOT=/path/to/DataSplit
```

各数据集目录包含 `images`、`masks`、`descriptions`；对应划分目录包含 `train.txt`、`val.txt`、`test.txt`，每行一个图片 ID。

## 训练

默认使用 ViT-L、400 轮、batch size 4、Adam、学习率 `1e-4`、余弦衰减和 BF16。输出磁盘建议至少保留 7 GiB 空闲空间。按需执行对应命令，`--output` 使用新的实验目录。

**NUAA-SIRST**

```bash
python -u main.py --mode train --dataset nuaa --data-root "$DATA_ROOT/NUAA-SIRST" --split-dir "$SPLIT_ROOT/NUAA" --output runs/nuaa
```

**NUDT-SIRST**

```bash
python -u main.py --mode train --dataset nudt --data-root "$DATA_ROOT/NUDT-SIRST" --split-dir "$SPLIT_ROOT/NUDT" --output runs/nudt
```

**IRSTD-1k**

```bash
python -u main.py --mode train --dataset irstd1k --data-root "$DATA_ROOT/IRSTD-1k" --split-dir "$SPLIT_ROOT/IRSTD-1k" --output runs/irstd1k
```

每个输出目录只保存验证集 IoU 最优的权重 `best_val.pt`，指标记录在 `training_metrics.txt`。下方推理使用 `best_val.pt`。

## 推理

将 `--weights` 替换为对应数据集的训练权重路径。

**NUAA-SIRST**

```bash
python -u main.py --mode infer --dataset nuaa --weights runs/nuaa/best_val.pt --data-root "$DATA_ROOT/NUAA-SIRST" --split-dir "$SPLIT_ROOT/NUAA" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```

**NUDT-SIRST**

```bash
python -u main.py --mode infer --dataset nudt --weights runs/nudt/best_val.pt --data-root "$DATA_ROOT/NUDT-SIRST" --split-dir "$SPLIT_ROOT/NUDT" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```

**IRSTD-1k**

```bash
python -u main.py --mode infer --dataset irstd1k --weights runs/irstd1k/best_val.pt --data-root "$DATA_ROOT/IRSTD-1k" --split-dir "$SPLIT_ROOT/IRSTD-1k" --split test --clip-checkpoint pretrained_checkpoint/ViT-B-32.pt
```


