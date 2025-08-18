import cv2
import numpy as np
import jittor as jt
from jittor import nn

from typing import Union

from basicsr.metrics.metric_util import reorder_image, to_y_channel


def calculate_psnr(
    img1: Union[np.ndarray, jt.Var],
    img2: Union[np.ndarray, jt.Var],
    crop_border: int,
    input_order: str = "HWC",
    test_y_channel: bool = False,
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    )
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )

    if isinstance(img1, jt.Var):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().numpy().transpose(1, 2, 0)
    if isinstance(img2, jt.Var):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_value = 1.0 if img1.max() <= 1 else 255.0
    return 20.0 * np.log10(max_value / np.sqrt(mse))


def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def prepare_for_ssim(img: np.ndarray, k: int):
    with jt.no_grad():
        img: jt.Var = jt.unsqueeze(jt.unsqueeze(jt.array(img), dim=0))
        img = img.float32()
        conv = nn.Conv(1, 1, k, stride=1, padding=k // 2, padding_mode="reflect")
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1.0 / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().numpy()


def prepare_for_ssim_rgb(img: np.ndarray, k: int):
    with jt.no_grad():
        img = jt.array(img).float32()  # HxWx3

        conv = nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode="reflect")
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1.0 / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(
                conv(img[:, :, i].unsqueeze(0).unsqueeze(0))
                .squeeze(0)
                .squeeze(0)[0::k, 0::k]
            )

    return jt.stack(new_img, dim=2).detach().numpy()


def _3d_gaussian_calculator(img: jt.Var, conv3d: nn.Conv3d) -> jt.Var:
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel() -> nn.Conv3d:
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = jt.array(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = nn.Conv3d(
        1,
        1,
        (11, 11, 11),
        stride=1,
        padding=(5, 5, 5),
        bias=False,
        padding_mode="replicate",
    )
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _ssim_3d(img1: np.ndarray, img2: np.ndarray, max_value: float):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel()

    img1 = jt.array(img1).float32()
    img2 = jt.array(img2).float32()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1**2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2**2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map: jt.Var = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def _ssim_cly(img1: np.ndarray, img2: np.ndarray) -> float:
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    crop_border: int,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    )
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )

    if isinstance(img1, jt.Var):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().numpy().transpose(1, 2, 0)
    if isinstance(img2, jt.Var):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim_cly(img1[..., 0], img2[..., 0])

    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with jt.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
    # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()
