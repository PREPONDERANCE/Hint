import numpy as np
import jittor as jt
from jittor import nn

from typing import Generator
from tqdm import tqdm
from scipy import linalg


@jt.no_grad()
def extract_inception_features(
    data_generator: Generator,
    inception: nn.Module,
    len_generator: int = None,
    device: str = "cuda",
):
    """Extract inception features.

    Args:
        data_generator (generator): A data generator.
        inception (nn.Module): Inception model.
        len_generator (int): Length of the data_generator to show the
            progressbar. Default: None.
        device (str): Device. Default: cuda.

    Returns:
        Tensor: Extracted features.
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit="batch", desc="Extract")
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        data = jt.to(data, device)
        feature = inception(data)[0].reshape(data.shape[0], -1)
        features.append(jt.to(feature, "cpu"))
    if pbar:
        pbar.close()
    features = jt.concat(features, 0)
    return features


def calculate_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
            representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, "Two mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Two covariances have different dimensions"

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print(
            "Product of cov matrices is singular. Adding {eps} to diagonal "
            "of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f"Imaginary component {m}")
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid
