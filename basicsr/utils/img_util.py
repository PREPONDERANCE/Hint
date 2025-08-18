import os
import cv2
import math
import numpy as np
import jittor as jt

from math import ceil
from typing import List, Union, Type, Tuple


@jt.no_grad()
def make_grid(
    tensor: jt.Var,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Tuple[int, int] = None,
    pad_value: float = 0.0,
) -> jt.Var:
    # ---- to numpy in CHW/NCHW ----
    arr = tensor.numpy()

    if arr.ndim == 2:  # (H, W) -> (1, 1, H, W)
        arr = arr[None, None, ...]
    elif arr.ndim == 3:  # (C, H, W) -> (1, C, H, W)
        arr = arr[None, ...]
    elif arr.ndim != 4:  # (N, C, H, W)
        raise ValueError("Expected 2D/3D/4D input (H W / C H W / N C H W).")

    N, C, H, W = arr.shape
    arr = arr.astype(np.float32, copy=False)

    # ---- normalize (optional, normalization) ----
    if normalize:
        if value_range is None:
            vmin, vmax = float(arr.min()), float(arr.max())
        else:
            vmin, vmax = value_range
        eps = 1e-7
        arr = (arr - vmin) / max(vmax - vmin, eps)

    # ---- grid layout ----
    xmaps = min(nrow, N)  # numbers per row
    ymaps = int(ceil(N / xmaps))  # total lines
    gh = ymaps * H + padding * (ymaps + 1)
    gw = xmaps * W + padding * (xmaps + 1)

    grid = np.full((C, gh, gw), pad_value, dtype=np.float32)

    k = 0
    y = padding
    for _ in range(ymaps):
        x = padding
        for _ in range(xmaps):
            if k >= N:
                break
            grid[:, y : y + H, x : x + W] = arr[k]
            x += W + padding
            k += 1
        y += H + padding

    return jt.array(grid)


def img2tensor(
    imgs: Union[List[np.ndarray], np.ndarray],
    bgr2rgb: bool = True,
    float32: bool = True,
) -> Union[List[jt.Var], jt.Var]:
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img: np.ndarray, bgr2rgb: bool, float32: bool):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: jt.Var = jt.array(img.transpose(2, 0, 1))
        if float32:
            img = img.float32()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(
    tensor: jt.Var,
    rgb2bgr: bool = True,
    out_type: Type = np.uint8,
    min_max: Tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """

    if not (
        jt.is_var(tensor)
        or (isinstance(tensor, list) and all(jt.is_var(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if jt.is_var(tensor):
        tensor = [tensor]

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float32().detach().clamp(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        _tensor: jt.Var
        n_dim = _tensor.ndim
        if n_dim == 4:
            # Fix from here
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}"
            )
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def imfrombytes(
    content: bytes,
    flag: str = "color",
    float32: bool = False,
) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    if img_np is None:
        raise Exception("None .. !!!")
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def imfrombytesDP(
    content: bytes,
    flag: str = "color",
    float32: bool = False,
) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    if img_np is None:
        raise Exception("None .. !!!")
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.0
    return img


def padding(img_lq: np.ndarray, img_gt: np.ndarray, gt_size: int):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt


def padding_DP(
    img_lqL: np.ndarray,
    img_lqR: np.ndarray,
    img_gt: np.ndarray,
    gt_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = img_gt.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lqL, img_lqR, img_gt

    img_lqL = cv2.copyMakeBorder(img_lqL, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lqR = cv2.copyMakeBorder(img_lqR, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lqL, img_lqR, img_gt


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: List[int] = None,
    auto_mkdir: bool = True,
) -> bool:
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def crop_border(
    imgs: Union[List[np.ndarray], np.ndarray],
    crop_border: int,
) -> List[np.ndarray]:
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
