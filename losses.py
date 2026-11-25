import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _align_logits_and_targets(inputs: torch.Tensor, targets: torch.Tensor):
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = (targets > 0).float()
    if inputs.shape[-2:] != targets.shape[-2:]:
        inputs = F.interpolate(
            inputs, size=targets.shape[-2:], mode="bilinear", align_corners=False
        )
    return (inputs, targets)


def _build_boundary_weight(
    targets: torch.Tensor, radius: int = 2, alpha: float = 4.0
) -> torch.Tensor:
    if radius <= 0 or alpha <= 0:
        return torch.ones_like(targets)
    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - targets, kernel_size=kernel_size, stride=1, padding=radius)
    boundary_band = (dilated - eroded).clamp(0.0, 1.0)
    return 1.0 + alpha * boundary_band


class SoftIoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-06):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        (logits, targets) = _align_logits_and_targets(logits, targets)
        probs = torch.sigmoid(logits.float())
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - iou).mean()


class BCELoss(nn.Module):
    def __init__(
        self, pos_weight: float = 3.0, boundary_radius: int = 2, boundary_alpha: float = 4.0
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.boundary_radius = boundary_radius
        self.boundary_alpha = boundary_alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        (logits, targets) = _align_logits_and_targets(logits, targets)
        weight_map = _build_boundary_weight(
            targets=targets, radius=self.boundary_radius, alpha=self.boundary_alpha
        )
        bce = F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        return (bce * weight_map).sum() / weight_map.sum().clamp_min(1.0)


class FinalOnlyLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-06,
        iou_weight: float = 1.0,
        bce_weight: float = 0.0,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.iou_fn = SoftIoULoss(smooth=smooth)
        self.bce_fn = BCELoss(pos_weight=pos_weight)

    def forward(self, final_masks: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_iou = self.iou_fn(final_masks, targets)
        loss_bce = self.bce_fn(final_masks, targets)
        loss_total = self.iou_weight * loss_iou + self.bce_weight * loss_bce
        return {"loss_total": loss_total, "loss_iou": loss_iou, "loss_bce": loss_bce}


def final_iou_loss(
    final_masks: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-06,
    iou_weight: float = 0.8,
    bce_weight: float = 0.2,
    pos_weight: float = 10.0,
) -> Dict[str, torch.Tensor]:
    loss_fn = FinalOnlyLoss(
        smooth=smooth, iou_weight=iou_weight, bce_weight=bce_weight, pos_weight=pos_weight
    )
    return loss_fn(final_masks, targets)


def compute_mmd_loss(text_tokens, image_tokens, bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0)):
    text_tokens = F.normalize(text_tokens, dim=-1)
    image_tokens = F.normalize(image_tokens, dim=-1)
    dist_xx = torch.cdist(text_tokens, text_tokens, p=2).pow(2)
    dist_yy = torch.cdist(image_tokens, image_tokens, p=2).pow(2)
    dist_xy = torch.cdist(text_tokens, image_tokens, p=2).pow(2)
    bandwidths = text_tokens.new_tensor(bandwidths).view(-1, 1, 1)
    kernel_xx = torch.exp(-dist_xx.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)
    kernel_yy = torch.exp(-dist_yy.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)
    kernel_xy = torch.exp(-dist_xy.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)
    return kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()


def is_lora_parameter_name(name: str) -> bool:
    return "lora_" in name or any(
        (x in name for x in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"))
    )


def set_trainable_by_policy(model):
    for (_, param) in model.named_parameters():
        param.requires_grad = False
    prompt_encoder_trainable_prefixes = (
        "prompt_encoder.mask_prompt_",
        "prompt_encoder.no_mask_embed",
        "prompt_encoder.sr_clip.text_adapter",
        "prompt_encoder.sr_clip.visual_adapter",
        "prompt_encoder.sr_clip.clip_backbone.scene_prompt_proj",
        "prompt_encoder.sr_clip.clip_backbone.projection",
        "prompt_encoder.sr_clip.clip_backbone.contexts",
        "prompt_encoder.sr_clip.clip_backbone.context_proj",
        "prompt_encoder.sr_clip.clip_backbone.visual_proj",
        "prompt_encoder.sr_clip.clip_backbone.visual.proj_visual",
        "prompt_encoder.sr_clip.clip_backbone.text_decoder",
        "prompt_encoder.sr_clip.clip_backbone.visual_decoder",
        "prompt_encoder.sr_clip.clip_backbone.scene_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.text_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.object_object_refine",
        "prompt_encoder.sr_clip.clip_backbone.scene_object_refine",
    )
    for (name, param) in model.named_parameters():
        if name.startswith("mask_decoder"):
            param.requires_grad = True
        if name.startswith(prompt_encoder_trainable_prefixes):
            param.requires_grad = True
        if "prompt_encoder.sr_clip.clip_backbone.transformer" in name and is_lora_parameter_name(
            name
        ):
            param.requires_grad = True
        if name.startswith("image_encoder") and is_lora_parameter_name(name):
            param.requires_grad = True
        if name.startswith("image_encoder.neck"):
            param.requires_grad = True
