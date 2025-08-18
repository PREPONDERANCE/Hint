import numpy as np
import jittor as jt

from jittor import nn

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ["none", "mean", "sum"]
device = "cuda" if jt.flags.use_cuda else "cpu"


@weighted_loss
def l1_loss(pred: jt.Var, target: jt.Var):
    return nn.smooth_l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred: jt.Var, target: jt.Var):
    return nn.mse_loss(pred, target, reduction="none")


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def execute(self, pred: jt.Var, target: jt.Var, weight: jt.Var = None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean"):
        super(FFTLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def fft2_as_ri(self, x: jt.Var) -> jt.Var:
        """
        x: real-valued jt.Var with shape (..., H, W) or (B, C, H, W)
        return: complex as real/imag in last dim, shape (..., H, W, 2)
        """
        # pack real -> [..., H, W, 2] = [real, imag]
        x_ri = jt.stack([x, jt.zeros_like(x)], dim=-1)

        # jt.nn._fft2 expects 4D with last dim=2: (N, H, W, 2)
        if x_ri.ndim == 3:  # (H, W, 2)
            x_ri = x_ri.unsqueeze(0)  # -> (1, H, W, 2)

        if x_ri.ndim == 5:  # (B, C, H, W, 2)
            b, c, h, w, _ = x_ri.shape
            y = jt.nn._fft2(x_ri.reshape(b * c, h, w, 2), inverse=False)
            return y.reshape(b, c, h, w, 2)

        elif x_ri.ndim == 4:  # (N, H, W, 2)  or (B, H, W, 2)
            return jt.nn._fft2(x_ri, inverse=False)

        else:
            raise ValueError("Expected 2D/3D/4D real input with spatial dims at the end.")

    def execute(self, pred: jt.Var, target: jt.Var, weight: jt.Var = None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """
        pred_fft = self.fft2_as_ri(pred)
        target_fft = self.fft2_as_ri(target)
        return self.loss_weight * l1_loss(
            pred_fft, target_fft, weight, reduction=self.reduction
        )


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def execute(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = jt.array([65.481, 128.553, 24.966]).reshape((1, 3, 1, 1))
        self.first = True

    def execute(self, pred: jt.Var, target: jt.Var):
        assert len(pred.shape) == 4

        if self.toY:
            if self.first:
                self.coef = jt.to(self.coef, device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.shape) == 4

        return (
            self.loss_weight
            * self.scale
            * jt.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean", eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def execute(self, x: jt.Var, y: jt.Var) -> jt.Var:
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = jt.mean(jt.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
