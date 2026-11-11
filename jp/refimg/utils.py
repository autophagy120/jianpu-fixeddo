import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def load_image(path: str, invert: bool = True) -> np.ndarray:
    """
    Loads an image from the given path and returns it as a numpy array.
    The image is loaded in grayscale mode and converted to uint8 data format.

    Args:
        path (str): The path to the image.
        invert (bool): Whether to invert the image. Default is True.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from path: {path}")
    
    image = image.astype(np.uint8)
    if invert:
        image = 255 - image  # invert the image
    
    return image

def save_image(image: np.ndarray, path: str, invert: bool = True) -> None:
    """
    Saves the given image to the given path.

    Args:
        image (np.ndarray): The image to save, float32 data format.
        path (str): The path to save the image to, converted to uint8 data format.
        invert (bool): Whether to invert the image. Default is True.
    """
    if invert:
        image = 1.0 - image  # invert the image
    
    # Ensure values are in [0, 255]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, image)

def get_image_stats(image: np.ndarray) -> str:
    """
    Returns a string containing the statistics of the given image.

    Args:
        image (np.ndarray): The image to get the statistics of.

    Returns:
        str: The statistics of the image.
    """
    stats = f"Image shape: {image.shape}\n"
    stats += f"Image dtype: {image.dtype}\n"
    stats += f"Image min: {image.min()}\n"
    stats += f"Image max: {image.max()}\n"
    stats += f"Image mean: {image.mean():.4f}\n"
    stats += f"Image median: {np.median(image):.4f}\n"
    stats += f"Image std: {image.std():.4f}"
    return stats

def crop_image(img: np.ndarray, x_offset: int, y_offset: int, x_size: int, y_size: int) -> np.ndarray:
    """
    根据给定的偏移量和大小裁剪图像，返回一个numpy二维数组。
    参数：
    img (numpy.ndarray): 图像数据
    x_offset (int): X轴偏移
    y_offset (int): Y轴偏移
    x_size (int): 宽度
    y_size (int): 高度
    
    返回：
    numpy.ndarray: 裁剪后的图像。
    """
    # 检查大小
    if img.shape[0] < y_offset + y_size or img.shape[1] < x_offset + x_size:
        raise ValueError('图像大小不足以裁剪')
    # 裁剪图像
    result_img = img[y_offset:y_offset+y_size, x_offset:x_offset+x_size]
    return result_img

def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_csv_data_with_timestamp(data: List[Tuple], output_dir: str, filename_prefix: str, headers: List[str]) -> str:
    """
    将数据保存为CSV文件, 文件名格式为 filename_prefix_YYYYMMDD_HHMMSS.csv
    
    Args:
        data (List[Tuple]): 要保存的数据列表
        output_dir (str): 输出目录
        filename_prefix (str): 文件名前缀
        headers (List[str]): CSV表头
    
    Returns:
        str: 保存的文件路径
    """
    ensure_dir(output_dir)
    time_signature = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{time_signature}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    
    return filepath

def plot_histogram(data: List[float], title: str, xlabel: str, output_path: str, 
                 min_val: Optional[float] = None, max_val: Optional[float] = None, bins: int = 20) -> None:
    """
    绘制直方图并保存
    
    Args:
        data (List[float]): 数据列表
        title (str): 图表标题
        xlabel (str): X轴标签
        output_path (str): 输出路径
        min_val (float, optional): 最小值
        max_val (float, optional): 最大值
        bins (int, optional): 直方图的bins数量
    """
    plt.figure(figsize=(10, 6))
    if min_val is not None and max_val is not None:
        plt.hist(data, bins=bins, range=(min_val, max_val))
    else:
        plt.hist(data, bins=bins)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_scatter(data: List[float], title: str, xlabel: str, ylabel: str, output_path: str) -> None:
    """
    绘制散点图并保存
    
    Args:
        data (List[float]): 数据列表
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        output_path (str): 输出路径
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, s=16, alpha=0.5, marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path, dpi=300)
    plt.close()

class ImageFileManager:
    """
    管理图像文件路径
    
    Attributes:
        image_paths (List[str]): 图像文件路径列表
    """
    
    def __init__(self, image_dir: str, image_exts: List[str] = ['.png', '.jpg', '.jpeg']):
        """
        初始化ImageFileManager，给入图像目录，以及图像文件扩展名列表，自动搜索目录下所有符合扩展名的文件
        
        Args:
            image_dir (str): 图像目录
            image_exts (List[str], optional): 图像文件扩展名列表. 默认为['.png', '.jpg', '.jpeg'].
        """
        self.image_paths = []
        for file_name in os.listdir(image_dir):
            if any(file_name.lower().endswith(ext.lower()) for ext in image_exts):
                self.image_paths.append(os.path.join(image_dir, file_name))
    
    def get_image_paths(self) -> List[str]:
        """
        获取图像文件路径列表
        
        Returns:
            List[str]: 图像文件路径列表
        """
        return self.image_paths 