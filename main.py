import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from checkpoint import atomic_save, tensor_storage_bytes
from dataset import SIRSTDataset, seed_worker, rows_from_ids, check_disjoint, train_pixel_stats
from lora import LoRA_sam, LoRA_CLIP_Text
from losses import final_iou_loss, compute_mmd_loss, set_trainable_by_policy
from metrics import Metrics
from segment_anything_training.build_sam import sam_model_registry

DATASETS = {"nuaa": "NUAA-SIRST", "nudt": "NUDT-SIRST", "irstd1k": "IRSTD-1k"}
SAM_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def duration(seconds):
    seconds = max(0, int(seconds))
    (hours, seconds) = divmod(seconds, 3600)
    (minutes, seconds) = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def rng_state():
    return dict(
        rng_python=random.getstate(),
        rng_numpy=np.random.get_state(),
        rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


@contextmanager
def training_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        if any(output.glob("*.pt*")) or (output / "metrics.jsonl").exists():
            raise FileExistsError(f"Existing results: {output}; choose another --output")
        yield
    finally:
        lock.unlink()


def setup(config, device):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config["seed"])
        if config["precision"] == "bf16" and (not torch.cuda.is_bf16_supported()):
            raise RuntimeError("BF16 is unsupported on this GPU")


def build_model(config, device, state=None):
    print(f"Initializing {config['model_type']} on {device}...", flush=True)
    os.environ["SAIST_CLIP_CHECKPOINT"] = str(Path(config["clip_checkpoint"]).resolve())
    with redirect_stdout(io.StringIO()):
        model = sam_model_registry[config["model_type"]](
            checkpoint=config["sam_checkpoint"] if state is None else None,
            pixel_mean=config["pixel_mean"],
            pixel_std=config["pixel_std"],
        )
        model = LoRA_sam(model, rank=config["sam_rank"], alpha=config["sam_alpha"]).sam
        LoRA_CLIP_Text(
            model.prompt_encoder.sr_clip.clip_backbone,
            rank=config["clip_rank"],
            alpha=config["clip_alpha"],
        )
        set_trainable_by_policy(model)
        model.image_encoder.gradient_checkpointing = config.get("gradient_checkpointing", True)
        if state is not None:
            model.load_state_dict(state, strict=True)
    return model.to(device)


def loader(rows, config, device, training=False):
    return DataLoader(
        SIRSTDataset(rows, training),
        batch_size=config["batch_size"] if training else config.get("eval_batch_size", 1),
        shuffle=training,
        drop_last=training,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )


def forward(model, batch, config, device):
    records = [
        dict(
            image=image.to(device, non_blocking=True),
            background=image.to(device, non_blocking=True),
            caption=caption.to(device, non_blocking=True),
            original_size=(1024, 1024),
        )
        for (image, caption) in zip(batch["image"], batch["caption"])
    ]
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and config["precision"] != "fp32",
    ):
        return model(records, multimask_output=False)


@torch.no_grad()
def evaluate(model, rows, config, device, split):
    model.eval()
    metrics = Metrics()
    batches = loader(rows, config, device)
    with tqdm(
        total=len(rows), desc=f"{config['dataset_name']} {split}", unit="img", dynamic_ncols=True
    ) as progress:
        for batch in batches:
            logits = forward(model, batch, config, device)[0].float()
            prediction = (logits.sigmoid() > 0.5).squeeze(1).cpu().numpy()
            truth = (batch["mask"] > 0.5).squeeze(1).numpy()
            if prediction.shape != truth.shape:
                raise ValueError("Prediction and mask shapes differ")
            for (pred, target) in zip(prediction, truth):
                metrics.update(pred, target)
            progress.update(len(prediction))
    values = metrics.get()
    return dict(split=split, IoU=values["IoU"], Pd_at_05=values["Pd"], Fa_at_05=values["Fa"])


def metric_line(config, row):
    prefix = f"epoch={row['epoch']} | " if "epoch" in row else ""
    return f"{prefix}[{config['dataset_name']} {row['split']}] IoU={100 * row['IoU']:.4f}% | Pd={100 * row['Pd_at_05']:.4f}% | Fa={1000000.0 * row['Fa_at_05']:.6f} (x1e-6)"


def select_best(best, results, epoch):
    improved = []
    for (split, row) in results.items():
        if not math.isfinite(row["IoU"]):
            raise FloatingPointError("Nonfinite IoU")
        if split not in best or row["IoU"] > best[split]["IoU"]:
            best[split] = dict(row, epoch=epoch)
            improved.append(split)
    return improved


def train(config, parts, device):
    output = Path(config["output"])
    setup(config, device)
    model = build_model(config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    checkpoint_bytes = tensor_storage_bytes(model.state_dict()) + 2 * sum(
        (p.numel() * p.element_size() for p in parameters)
    )
    required = int(3.15 * checkpoint_bytes) + 192 * 1024**2
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(f"Need {required / 1024 ** 3:.2f} GiB free, have {free / 1024 ** 3:.2f} GiB")
    optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config["precision"] == "fp16"
    )
    print("Trainable parameters:", sum((p.numel() for p in parameters)), flush=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (best, timings) = ({}, [])
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        begun = time.perf_counter()
        model.train()
        lr = config["min_lr"] + (config["lr"] - config["min_lr"]) * 0.5 * (
            1 + math.cos(math.pi * (epoch - 1) / max(1, config["epochs"] - 1))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        total = 0.0
        batches = loader(parts["train"], config, device, training=True)
        print(f"epoch={epoch}/{config['epochs']} started; batches={len(batches)}", flush=True)
        for (step, batch) in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            (logits, text, image, _) = forward(model, batch, config, device)
            target = batch["mask"].to(device, non_blocking=True)
            loss = final_iou_loss(
                logits,
                target,
                iou_weight=config["iou_weight"],
                bce_weight=config["bce_weight"],
                pos_weight=config["pos_weight"],
            )["loss_total"]
            loss = loss + config["mmd_weight"] * compute_mmd_loss(text.float(), image.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            if step % 10 == 0 or step + 1 == len(batches):
                elapsed = time.perf_counter() - begun
                eta = elapsed / (step + 1) * (len(batches) - step - 1)
                print(
                    f"epoch={epoch}/{config['epochs']} batch={step + 1}/{len(batches)} loss={loss.item():.6f} lr={lr:.8g} elapsed={duration(elapsed)} epoch_train_ETA={duration(eta)}",
                    flush=True,
                )
        results = {}
        if epoch == 1 or epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
            results["val"] = evaluate(model, parts["val"], config, device, "val")
        with preserve_rng():
            results["test"] = evaluate(model, parts["test"], config, device, "test")
        improved = select_best(best, results, epoch)
        for split in improved:
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                config=config,
                best=best[split]["IoU"],
                best_metrics=best,
                selected_by=f"{split}_IoU",
                selected_metrics=best[split],
                evaluation_metrics={
                    key: dict(value, epoch=epoch) for (key, value) in results.items()
                },
                **rng_state(),
            )
            atomic_save(state, output / f"best_{split}.pt")
        temporary = output / "best_metrics.json.tmp"
        temporary.write_text(
            json.dumps(dict(best=best), indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, output / "best_metrics.json")
        timings.append(time.perf_counter() - begun)
        eta = sum(timings[-5:]) / len(timings[-5:]) * (config["epochs"] - epoch)
        timing = f"epoch_time={duration(timings[-1])} elapsed={duration(time.perf_counter() - started)} ETA={duration(eta)}"
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as json_file, (
            output / "training_metrics.txt"
        ).open("a", encoding="utf-8") as text_file:
            for values in results.values():
                row = dict(values, epoch=epoch, loss=total / len(batches), lr=lr)
                json_file.write(json.dumps(row, allow_nan=False) + "\n")
                line = metric_line(config, row) + " | " + timing
                text_file.write(line + "\n")
                print(line, flush=True)
    for split in ("val", "test"):
        print(f"Best {split} checkpoint: {output / ('best_' + split + '.pt')}", flush=True)


def training_config(args):
    stats = {
        "nuaa": ([109.34], [57.02]),
        "nudt": ([107.02], [55.56]),
        "irstd1k": ([87.43570035044353], [57.54670695075356]),
    }
    (mean, std) = stats[args.dataset]
    sam = Path(
        args.sam_checkpoint or "pretrained_checkpoint/" + SAM_FILES[args.model_type]
    ).resolve()
    clip = Path(args.clip_checkpoint or "pretrained_checkpoint/ViT-B-32.pt").resolve()
    for (variant, filename) in SAM_FILES.items():
        if sam.name == filename and variant != args.model_type:
            raise ValueError("SAM checkpoint and model type differ")
    for path in (sam, clip):
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(
        dataset_name=DATASETS[args.dataset],
        model_type=args.model_type,
        sam_checkpoint=str(sam),
        clip_checkpoint=str(clip),
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=1,
        workers=args.workers if args.workers is not None else 4,
        lr=args.lr,
        min_lr=1e-06,
        weight_decay=0.0,
        seed=args.seed,
        precision=args.precision or "bf16",
        eval_interval=5,
        sam_rank=2,
        sam_alpha=4.0,
        clip_rank=2,
        clip_alpha=4.0,
        iou_weight=0.8,
        bce_weight=0.2,
        pos_weight=10.0,
        mmd_weight=0.05,
        gradient_checkpointing=True,
        pixel_mean=mean,
        pixel_std=std,
        data_root=str(Path(args.data_root).resolve()),
        split_dir=str(Path(args.split_dir).resolve()),
        output=str(Path(args.output or "runs/" + args.dataset).resolve()),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--model-type", choices=tuple(SAM_FILES), default="vit_l")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or (args.workers is not None and args.workers < 0):
        parser.error("Require epochs >= 1, batch-size >= 2 and workers >= 0")
    if not math.isfinite(args.lr) or args.lr < 1e-06:
        parser.error("lr must be finite and >= 1e-6")
    if args.mode == "train" and args.weights:
        parser.error("--weights requires --mode infer")
    device = torch.device(args.device)
    if args.mode == "train":
        config = training_config(args)
        parts = {
            split: rows_from_ids(args.data_root, Path(args.split_dir) / f"{split}.txt")
            for split in ("train", "val", "test")
        }
        check_disjoint(*parts.values())
        if len(parts["train"]) < config["batch_size"]:
            raise ValueError("Training set is smaller than batch-size")
        if args.dataset == "irstd1k":
            (config["pixel_mean"], config["pixel_std"]) = train_pixel_stats(parts["train"])
        with training_lock(Path(config["output"])):
            train(config, parts, device)
    else:
        if not args.weights:
            parser.error("--mode infer requires --weights")
        print(f"Loading checkpoint: {args.weights}", flush=True)
        saved = torch.load(args.weights, map_location="cpu", weights_only=False)
        if (
            not isinstance(saved, dict)
            or not isinstance(saved.get("config"), dict)
            or "model" not in saved
        ):
            raise ValueError("Checkpoint must contain model and config")
        config = dict(saved["config"])
        if config.get("dataset_name") != DATASETS[args.dataset]:
            raise ValueError("Checkpoint and --dataset differ")
        config["workers"] = args.workers if args.workers is not None else 0
        config["precision"] = args.precision or config.get("precision", "fp32")
        config["clip_checkpoint"] = str(
            Path(args.clip_checkpoint or config["clip_checkpoint"]).resolve()
        )
        if not Path(config["clip_checkpoint"]).is_file():
            raise FileNotFoundError(config["clip_checkpoint"])
        rows = rows_from_ids(args.data_root, Path(args.split_dir) / f"{args.split}.txt")
        setup(config, device)
        model = build_model(config, device, saved["model"])
        del saved
        print(metric_line(config, evaluate(model, rows, config, device, args.split)), flush=True)


if __name__ == "__main__":
    main()
