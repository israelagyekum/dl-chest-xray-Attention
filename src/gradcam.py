"""
gradcam.py — Vanilla Grad-CAM for the baseline models.

Usage:
    gcam = GradCAM(model, target_layer_name="backbone.features.7")
    cam  = gcam(x, class_idx=None)   # (B, H, W) upsampled to input resolution

GradCAM is applied to BaselineModel checkpoints to provide the
post-hoc localisation baseline we compare our supervised attention against.

For a fair comparison, the CAM is upsampled to the same grid resolution as
the attention map (7×7 or 14×14) before computing IoU / pointing-game.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization," ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Hook-based Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM using forward/backward hooks on a named layer.

    Parameters
    ----------
    model            : nn.Module — the model (BaselineModel or AttentionModel)
    target_layer_name: str  — dot-separated attribute path, e.g.
                              "backbone.features.7"  (ResNet50 last conv block)
                              "backbone.features.denseblock4"  (DenseNet121)
                              "backbone.features.8"  (EfficientNet-B0)
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._handles = []

        # Resolve the target layer from the dot-path
        target = self._get_layer(model, target_layer_name)

        # Forward hook: capture activations
        self._handles.append(
            target.register_forward_hook(self._save_activation)
        )
        # Backward hook: capture gradients
        self._handles.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    # ---- Hook callbacks ------------------------------------------------

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ---- Helper --------------------------------------------------------

    @staticmethod
    def _get_layer(model: nn.Module, layer_path: str) -> nn.Module:
        parts = layer_path.split(".")
        m = model
        for p in parts:
            if p.isdigit():
                m = list(m.children())[int(p)]
            else:
                m = getattr(m, p)
        return m

    # ---- Main API ------------------------------------------------------

    def __call__(self,
                 x:          torch.Tensor,
                 class_idx:  Optional[Union[int, torch.Tensor]] = None,
                 output_size: Optional[tuple] = None
                 ) -> np.ndarray:
        """
        Compute Grad-CAM heatmaps.

        Parameters
        ----------
        x           : (B, 3, H, W) input tensor
        class_idx   : int or (B,) tensor of class indices.
                      If None, uses the max-probability class per image.
        output_size : (H_out, W_out) to upsample CAM to.
                      If None, returns at the feature-map resolution.

        Returns
        -------
        cams : np.ndarray (B, H_out, W_out) — normalised to [0, 1]
        """
        self.model.zero_grad()
        logits, _ = self.model(x)           # (B, 14), _

        B = logits.size(0)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)  # (B,)
        elif isinstance(class_idx, int):
            class_idx = torch.full((B,), class_idx,
                                   dtype=torch.long, device=logits.device)

        # Scalar to differentiate: sum of target class logits over batch
        target_scores = logits[range(B), class_idx].sum()
        target_scores.backward()

        # Global-average pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1)         # (B, G, G)
        cam     = F.relu(cam)                                       # keep positives

        # Normalise per image
        B_, G_, _ = cam.shape
        cam_flat  = cam.view(B_, -1)
        cam_min   = cam_flat.min(dim=1, keepdim=True).values.view(B_, 1, 1)
        cam_max   = cam_flat.max(dim=1, keepdim=True).values.view(B_, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)         # (B, G, G)

        # Upsample if requested
        if output_size is not None:
            cam = F.interpolate(
                cam.unsqueeze(1), size=output_size, mode="bilinear",
                align_corners=False
            ).squeeze(1)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# Layer name helpers per backbone
# ---------------------------------------------------------------------------

GRADCAM_LAYERS = {
    "resnet50":        "backbone.features.7",        # layer4 (last conv block)
    "densenet121":     "backbone.features.norm5",    # final BN after denseblock4
    "efficientnet_b0": "backbone.features.8",        # last conv block
}


def get_gradcam_layer(backbone_name: str) -> str:
    """Return the default Grad-CAM target layer for a given backbone."""
    return GRADCAM_LAYERS.get(backbone_name.lower(), "backbone.features")


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def compute_gradcam_batch(model:          nn.Module,
                           loader:         torch.utils.data.DataLoader,
                           backbone_name:  str,
                           device:         str  = "cpu",
                           grid_size:      int  = 7,
                           class_idx:      Optional[int] = None,
                           max_batches:    Optional[int] = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Grad-CAM over an entire DataLoader.

    Returns
    -------
    all_cams     : (N, grid_size, grid_size)  — normalised Grad-CAM maps
    all_masks    : (N, grid_size, grid_size)  — gt bounding-box masks
    all_has_box  : (N,)                       — has-box flags
    """
    layer_name = get_gradcam_layer(backbone_name)
    gcam       = GradCAM(model, layer_name)
    model.to(device)

    cams_list    = []
    masks_list   = []
    has_box_list = []

    for batch_idx, (imgs, labels, masks, has_box) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = imgs.to(device)

        cam = gcam(imgs, class_idx=class_idx,
                   output_size=(grid_size, grid_size))  # (B, G, G)
        cams_list.append(cam)
        masks_list.append(masks.numpy())
        has_box_list.append(has_box.numpy())

    gcam.remove_hooks()

    return (
        np.concatenate(cams_list,    axis=0),
        np.concatenate(masks_list,   axis=0),
        np.concatenate(has_box_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.models.model import BaselineModel

    model = BaselineModel("resnet50", pretrained=False)
    layer = get_gradcam_layer("resnet50")
    gcam  = GradCAM(model, layer)

    x    = torch.randn(2, 3, 224, 224)
    cams = gcam(x, class_idx=0, output_size=(7, 7))
    print(f"CAM shape : {cams.shape}")
    print(f"CAM range : [{cams.min():.3f}, {cams.max():.3f}]")
    assert cams.min() >= 0.0 and cams.max() <= 1.0 + 1e-6
    gcam.remove_hooks()
    print("✓ Grad-CAM smoke-test passed.")
