import numpy as np
from typing import Tuple, Optional

from skimage.filters import threshold_otsu

import logging
logger = logging.getLogger(__name__)

# def estimate_background_foreground_values(image: np.ndarray, bg_quantile: float = 0.5, 
#                                          fg_count: int = 1024) -> Tuple[float, float]:
#     """
#     Estimate the global background (low-intensity) and foreground (high-intensity) values of a grayscale image.

#     Parameters
#     ----------
#     image : np.ndarray
#         2D 或 3D 灰度图，dtype 必须为 uint8。
#     bg_quantile : float
#         背景分位点 α，取值范围 [0,1]，默认 0.5（中值以下）。
#     fg_count : int
#         前景像素数量，默认 1024

#     Returns
#     -------
#     Tuple[float, float]
#         全局背景值和全局前景值。
#     """
#     # 强制为 uint8
#     if image.dtype != np.uint8:
#         raise ValueError("Image dtype should be uint8.")
#     # 防止 quantile 越界
#     bg_alpha = float(np.clip(bg_quantile, 0.0, 1.0))

#     # 展平
#     flat = image.ravel()
#     # 统计直方图（各个像素值的频数）
#     hist = np.bincount(flat, minlength=256)
#     # 计算累积分布
#     cdf = hist.cumsum()
#     # 计算总像素数
#     total = cdf[-1]

#     # 找到背景
#     # 找到对应分位的灰度阈值
#     bg_under_value = np.searchsorted(cdf, bg_alpha * total)
#     # 直接用直方图算加权平均
#     hist_bg = hist[:bg_under_value + 1]
#     if hist_bg.sum() == 0:
#         # 图像全黑或无值
#         return float(bg_under_value), 255.0
#     weights = np.arange(bg_under_value + 1)
#     weighted_sum = (weights * hist_bg).sum()
#     bg_estimate = weighted_sum / hist_bg.sum()

#     # 找到前景
#     fp_above_value = np.searchsorted(cdf, total - fg_count)
#     hist_fp = hist[fp_above_value:]
#     if hist_fp.sum() == 0:
#         # 图像全黑或无值
#         return float(bg_estimate), float(fp_above_value)
#     weights = np.arange(fp_above_value, 256)
#     weighted_sum = (weights * hist_fp).sum()
#     fp_estimate = weighted_sum / hist_fp.sum()

#     return bg_estimate, fp_estimate

def estimate_background_values(image: np.ndarray, bg_quantile: float = 0.5) -> float:
    """
    Estimate the global background (low-intensity) values of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D 或 3D 灰度图，dtype 必须为 uint8。
    bg_quantile : float
        背景分位点 α，取值范围 [0,1]，默认 0.5（中值以下）。

    Returns
    -------
    float
        全局背景值。
    """
    # 强制为 uint8
    if image.dtype != np.uint8:
        raise ValueError("Image dtype should be uint8.")
    # 防止 quantile 越界
    bg_alpha = float(np.clip(bg_quantile, 0.0, 1.0))

    # 展平
    flat = image.ravel()
    # 统计直方图（各个像素值的频数）
    hist = np.bincount(flat, minlength=256)
    # 计算累积分布
    cdf = hist.cumsum()
    # 计算总像素数
    total = cdf[-1]

    # 找到背景
    # 找到对应分位的灰度阈值
    bg_under_value = np.searchsorted(cdf, bg_alpha * total)
    # 直接用直方图算加权平均
    hist_bg = hist[:bg_under_value + 1]
    if hist_bg.sum() == 0:
        return float(bg_under_value)
    weights = np.arange(bg_under_value + 1)
    weighted_sum = (weights * hist_bg).sum()
    bg_estimate = weighted_sum / hist_bg.sum()

    return bg_estimate

def estimate_foreground_values(image: np.ndarray) -> float:
    """
    Estimate the global foreground (high-intensity) values of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D 或 3D 灰度图，dtype 必须为 float32。

    Returns
    -------
    float
        全局前景值。
    """

    # 通过 Otsu 阈值得到前景阈值
    fg_above = threshold_otsu(image)
    # 统计前景部分的平均值
    fg_mean = image[image > fg_above].mean()
    return fg_mean

# def preprocess_image(image: np.ndarray, bg_quantile: float = 0.8, fg_count: int = 256, 
#                     bg_maps_to: float = 0.01, fg_maps_to: float = 0.99, 
#                     eps: float = 1e-8, debug: bool = False) -> np.ndarray:
#     """
#     Preprocesses the image by applying adaptive lighting correction
    
#     Args:
#         image (np.ndarray): Input image in uint8 format
#         bg_quantile (float): Background quantile
#         fg_count (int): Number of foreground pixels to consider
#         bg_maps_to (float): Target value for background
#         fg_maps_to (float): Target value for foreground
#         eps (float): Small epsilon to prevent division by zero
#         debug (bool): Whether to print debug information
        
#     Returns:
#         np.ndarray: Preprocessed image in float32 format [0,1]
#     """
#     from . import utils  # Import here to avoid circular imports

#     if debug:
#         print("Before preprocessing:" + utils.get_image_stats(image))

#     # estimate background and foreground values
#     bg_estimate, fg_estimate = estimate_background_foreground_values(image, bg_quantile, fg_count)
#     if debug:
#         print(f"calculated bg_estimate: {bg_estimate} (0~255), calculated fg_estimate: {fg_estimate} (0~255)")

#     # cast to float32
#     image = image.astype(np.float32) / 255.0
#     low_value = image.min()
#     high_value = image.max()
#     if debug:
#         print(f"image low_value: {low_value}, image high_value: {high_value}")

#     # adjust image range to [0,1]
#     image = (image - low_value) / (high_value - low_value + eps)
#     bg_estimate = (float(bg_estimate) / 255.0 - low_value) / (high_value - low_value + eps)
#     fg_estimate = (float(fg_estimate) / 255.0 - low_value) / (high_value - low_value + eps)

#     # bg gamma correction
#     # to let bg_estimate ** gamma = bg_maps_to, we need to find gamma
#     gamma = np.log(bg_maps_to + eps) / np.log(bg_estimate + eps)
#     image = np.power(image, gamma)
#     image = np.clip(image, 0.0, 1.0)

#     threshold = threshold_otsu(image)

#     # fg gamma correction
#     # to let fg_estimate ** gamma = fg_maps_to (inverted), we need to find gamma
#     # image = 1.0 - image
#     # gamma = np.log(1.0 - fg_maps_to + eps) / np.log(1.0 - fg_estimate + eps)
#     # # apply gamma correction
#     # image = np.power(image, gamma)
#     # image = 1.0 - image

#     if debug:
#         print("After preprocessing:" + utils.get_image_stats(image))
    
#     return image 

def adaptive_lighting_correction(image: np.ndarray, bg_quantile: float = 0.8, 
                    bg_maps_to: float = 0.01, fg_maps_to: float = 0.9,
                    use_safe_gammas: bool = True, safe_gamma_range: Tuple[float, float] = (0.25, 4.0),
                    eps: float = 1e-8, debug: bool = False, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Preprocesses the image by applying adaptive lighting correction
    
    Args:
        image (np.ndarray): Input image in uint8 format
        bg_quantile (float): Background quantile
        bg_maps_to (float): Target value for background
        use_safe_gammas (bool): Whether to use safe gamma range
        safe_gamma_range (Tuple[float, float]): Safe gamma range
        eps (float): Small epsilon to prevent division by zero
        debug (bool): Whether to print debug information
        
    Returns:
        np.ndarray: Preprocessed image in float32 format [0,1]
        dict: Log information, including low_value, high_value, gamma_bg_correction, gamma_fg_correction
    """
    from . import utils  # Import here to avoid circular imports

    if debug:
        print("Before preprocessing:" + utils.get_image_stats(image))

    # estimate background and foreground values
    bg_estimate = estimate_background_values(image, bg_quantile)
    if debug:
        print(f"calculated bg_estimate: {bg_estimate} (0~255)")

    # cast to float32
    image = image.astype(np.float32) / 255.0
    raw_low_value = image.min()
    raw_high_value = image.max()
    if debug:
        print(f"image low_value: {raw_low_value}, image high_value: {raw_high_value}")

    # 【A】：输出原始图像及其灰度直方图，在直方图中以竖线标记出 bg_estimate，raw_low_value，raw_high_value 的位置，同时标记出预测 bg_estimate 时所用的 背景分位点 α 的位置。

    # adjust image range to [0,1]
    image = (image - raw_low_value) / (raw_high_value - raw_low_value + eps)
    bg_estimate = (float(bg_estimate) / 255.0 - raw_low_value) / (raw_high_value - raw_low_value + eps)

    logger = logger or logging.getLogger(__name__)

    # 【B】：输出此处色彩标准化后图像及其灰度直方图，在直方图中以竖线标记出变换后 bg_estimate， raw_low_value（应为 0），raw_high_value（应为 1）的位置，并在直方图中画出将要进行的第一次 gamma 变换曲线。

    # bg gamma correction
    # to let bg_estimate ** gamma = bg_maps_to, we need to find gamma
    gamma_bg_correction = np.log(bg_maps_to + eps) / np.log(bg_estimate + eps)
    if use_safe_gammas:
        if gamma_bg_correction < safe_gamma_range[0]:
            gamma_bg_correction = safe_gamma_range[0]
            logger.warning("gamma_bg_correction is too small, restricted to safe_gamma setting")
        elif gamma_bg_correction > safe_gamma_range[1]:
            gamma_bg_correction = safe_gamma_range[1]
            logger.warning("gamma_bg_correction is too large, restricted to safe_gamma setting")
    image = np.power(image, gamma_bg_correction)
    image = np.clip(image, 0.0, 1.0)

    # 【C】：输出此处第一次 gamma 矫正后图像及其灰度直方图，在直方图中以竖线标记出变换后 bg_estimate 的位置。

    fg_estimate = estimate_foreground_values(image)
    if debug:
        print(f"calculated fg_estimate: {fg_estimate} (0~1)")

    # 【D】：输出此处同上图像及其灰度直方图，以及新预测的 fg_estimate 的位置，同时标记出预测 fg_estimate 时所用的 Otsu 阈值的位置，并在直方图中画出将要进行的第二次 gamma 变换曲线。

    # fg gamma correction
    # to let fg_estimate ** gamma = fg_maps_to (inverted), we need to find gamma
    image = 1.0 - image
    gamma_fg_correction = np.log(1.0 - fg_maps_to + eps) / np.log(1.0 - fg_estimate + eps)
    if use_safe_gammas:
        if gamma_fg_correction < safe_gamma_range[0]:
            gamma_fg_correction = safe_gamma_range[0]
            logger.warning("gamma_fg_correction is too small, restricted to safe_gamma setting")
        elif gamma_fg_correction > safe_gamma_range[1]:
            gamma_fg_correction = safe_gamma_range[1]
            logger.warning("gamma_fg_correction is too large, restricted to safe_gamma setting")
    # apply gamma correction
    image = np.power(image, gamma_fg_correction)
    image = 1.0 - image
    image = np.clip(image, 0.0, 1.0)

    # 【E】：输出此处第二次 gamma 矫正后图像及其灰度直方图，在直方图中标记出变换后 fg_estimate 的位置。

    if debug:
        print("After preprocessing:" + utils.get_image_stats(image))

    log = {
        "raw_low_value": raw_low_value,
        "raw_high_value": raw_high_value,
        "gamma_bg_correction": gamma_bg_correction,
        "gamma_fg_correction": gamma_fg_correction
    }
    
    return image, log