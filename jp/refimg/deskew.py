import numpy as np
import scipy.ndimage

from typing import Optional

def generate_shear_kernel(image_width: int, theta_deg: float) -> np.ndarray:
    """
    生成斜切变换的卷积核。
    
    参数：
    image_width (int): 图像的宽度（横向像素数）。
    theta_deg (float): 旋转角度，以度为单位。
    
    返回：
    numpy.ndarray: 斜切变换的卷积核，高度保证为奇数，宽度为 image_width。
    """
    
    is_theta_negative = theta_deg < 0
    if is_theta_negative:
        theta_deg = -theta_deg

    if type(image_width) != int or image_width <= 0:
        raise ValueError("image_width must be a positive integer.")

    # 计算每个像素的横向位置 x[j]
    if image_width % 2 == 0:
        # 偶数宽度
        x = np.arange(0, image_width // 2) + 0.5
    else:
        # 奇数宽度
        x = np.arange(0, image_width // 2 + 1)

    # 计算每个位置的纵向偏移量 Δy[j] = x[j] * tan(θ)
    delta_y = x * np.tan(np.radians(theta_deg))
    
    # 计算每个位置的上、下两个点的偏移量
    delta_u = np.floor(delta_y).astype(int)
    delta_l = (delta_u + 1).astype(int)

    # 初始化右下侧的卷积核h[i, j]
    quarter_h = np.zeros((delta_l[-1] + 1, x.shape[0]))

    # 计算每个位置的上、下两个点的卷积权重
    quarter_h[delta_u, np.arange(x.shape[0])] = delta_l - delta_y
    quarter_h[delta_l, np.arange(x.shape[0])] = delta_y - delta_u

    # 从四分之一卷积核得到完整卷积核
    h = np.zeros((2 * delta_l[-1] + 1, image_width))
    if image_width % 2 == 0:
        # 偶数宽度
        # 右下直接复制
        h[delta_l[-1]:, image_width//2:] = quarter_h
        # 左上翻转复制
        h[:delta_l[-1]+1, :image_width//2] = quarter_h[::-1, ::-1]
    else:
        # 奇数宽度
        # 右下直接复制
        h[delta_l[-1]:, image_width//2:] = quarter_h
        # 左上翻转复制，记得不要重复中间一排位置
        h[:delta_l[-1]+1, :image_width//2] = quarter_h[::-1, :0:-1]

    if is_theta_negative:
        # 若旋转角度为负，则对卷积核进行镜像
        h = h[::-1, :]
    
    return h

def perform_shear_transform(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    使用卷积进行斜切变换，得到变换后的图像g。
    
    Args:
        f (np.ndarray): 原图像(二维数组)
        h (np.ndarray): 斜切变换卷积核(二维数组)
    
    Returns:
        np.ndarray: 斜切变换后的图像
    """
    assert f.shape[1] == h.shape[1]

    g = np.zeros(f.shape)
    for j in range(f.shape[1]):
        g[:, j] = scipy.ndimage.convolve(f[:, j], h[:, j], mode='reflect')
    
    return g

def compute_entropy(g: np.ndarray, p_delta: float = 1e-8) -> float:
    """
    计算行水平投影分布的香农熵，使用平滑处理以避免对数零值。
    
    Args:
        g (np.ndarray): 输入图像
        p_delta (float): 拉普拉斯平滑参数，避免 p[i] 为零时对数为负无穷
    
    Returns:
        float: 香农熵值
    """
    b = np.sum(g, axis=1) # 计算水平投影
    p = b / np.sum(b) + p_delta  # 拉普拉斯平滑
    p = p / np.sum(p)  # 再次归一化
    entropy = -np.sum(p * np.log2(p)) / np.log2(p.size)  # 使用log2计算香农熵
    return entropy

def solve_H(f: np.ndarray, theta: float) -> float:
    """
    计算给定角度下图像的熵值
    
    Args:
        f (np.ndarray): 输入图像
        theta (float): 旋转角度，单位为度
        
    Returns:
        float: 熵值
    """
    image_width = f.shape[1]
    # a. Generate shear kernel
    shear_kernel = generate_shear_kernel(image_width, theta)
    # b. Perform shear transform
    g = perform_shear_transform(f, shear_kernel)
    # c. Compute entropy of transformed image
    H_theta = compute_entropy(g)
    return H_theta

def resize_with_gaussian(f: np.ndarray, n: int = 2, blur_sigma_factor: float = 0.8) -> np.ndarray:
    """
    将二维图像 f 按比例收缩为原来的 1/n，采用高斯模糊后直接采样。
    
    Args:
        f (np.ndarray): 输入的二维图像，形状为 (H, W)
        n (int): 收缩比例的倒数，表示缩小为原来的 1/n
        blur_sigma_factor (float): 表示高斯模糊的标准差是 n 的多少倍，控制模糊的强度
    
    Returns:
        np.ndarray: 收缩后的二维图像
    """
    # 应用高斯模糊
    f_blurred = scipy.ndimage.gaussian_filter(f, sigma=blur_sigma_factor*n)
    # 使用直接采样
    resized_f = f_blurred[::n, ::n]
    return resized_f

def golden_section_search(func, a: float, b: float, iterations: int = 5) -> tuple:
    """
    使用黄金分割法寻找函数的最小值区间
    
    Args:
        func: 目标函数
        a (float): 区间下界
        b (float): 区间上界
        iterations (int): 迭代次数
    
    Returns:
        tuple: 最终的区间 (a, b)
    """
    golden_ratio = (1 + np.sqrt(5)) / 2  # 黄金比例
    
    # 计算两个分割点
    x1 = b - (b - a) / golden_ratio
    x2 = a + (b - a) / golden_ratio
    
    # 计算对应的函数值
    f_x1 = func(x1)
    f_x2 = func(x2)
    
    for i in range(iterations):  # 根据迭代次数进行多次迭代
        if f_x1 < f_x2:
            b = x2
            if i < iterations - 1: # 最后一次迭代不再计算新的分割点
                x2 = x1
                f_x2 = f_x1
                x1 = b - (b - a) / golden_ratio
                f_x1 = func(x1)
        else:
            a = x1
            if i < iterations - 1: # 最后一次迭代不再计算新的分割点
                x1 = x2
                f_x1 = f_x2
                x2 = a + (b - a) / golden_ratio
                f_x2 = func(x2)
    
    # 返回区间 a 和 b
    return a, b

def search_with_order_iterations(img: np.ndarray, init_a: float, init_b: float, 
                               num_iterations: list = [4, 4, 4]) -> float:
    """
    多层次搜索最优旋转角度
    
    Args:
        img (np.ndarray): 输入图像
        init_a (float): 初始区间下界
        init_b (float): 初始区间上界
        num_iterations (list): 每层次的迭代次数
    
    Returns:
        float: 最优旋转角度
    """
    # 定义不同阶次的函数及对应的迭代次数
    order_info = []
    current_img = img
    for i, num_iter in enumerate(num_iterations):
        order_info.append({
            "img": current_img,
            "num_iter": num_iter,
        })
        if i < len(num_iterations) - 1:
            current_img = resize_with_gaussian(current_img)

    a = init_a
    b = init_b
    # 依次处理不同阶次的函数，但是记得顺序要反过来（从阶次高的开始向 0 来搜索）
    for item in order_info[::-1]:
        a, b = golden_section_search(lambda theta: solve_H(item['img'], theta), a, b, iterations=item["num_iter"])
    
    return (a + b) / 2  # 返回最优角度

# def grey_mapping(img: np.ndarray) -> np.ndarray:
#     """
#     使用 np.piecewise 优化灰度变换的批量处理方法。
    
#     Args:
#         img (np.ndarray): 输入图像，灰度值在0到1之间。

#     Returns:
#         np.ndarray: 变换后的图像。
#     """
#     # 定义各个区间的映射
#     conditions = [
#         (img <= 0.1),
#         (img > 0.1) & (img <= 0.3),
#         (img > 0.3) & (img <= 0.5),
#         (img > 0.5) & (img <= 0.7),
#         (img > 0.7) & (img <= 0.9),
#         (img > 0.9)
#     ]
    
#     # 对应条件下的输出值
#     values = [
#         0.0,
#         lambda x: (x - 0.1) * (0.1 - 0.0) / (0.3 - 0.1) + 0.0,
#         lambda x: (x - 0.3) * (0.6 - 0.1) / (0.5 - 0.3) + 0.1,
#         lambda x: (x - 0.5) * (0.9 - 0.6) / (0.7 - 0.5) + 0.6,
#         lambda x: (x - 0.7) * (1.0 - 0.9) / (0.9 - 0.7) + 0.9,
#         1.0
#     ]
    
#     # 使用 np.piecewise 进行变换
#     new_img = np.piecewise(img, conditions, values)
    
#     return new_img

def deskew_image(image: np.ndarray, min_angle: float = -5, max_angle: float = 5, 
                num_iterations: list = [7, 5, 3], debug: bool = False) -> tuple:
    """
    对图像进行旋转矫正
    
    Args:
        image (np.ndarray): 输入图像 (已经进行过光照校正的浮点型图像, 范围[0,1])
        min_angle (float): 搜索角度下界
        max_angle (float): 搜索角度上界
        num_iterations (list): 多层次搜索迭代次数
        debug (bool): 是否打印调试信息
        
    Returns:
        tuple: (处理后的图像, 预测的倾斜角度)
    """
    
    # 预测倾斜角度
    pred_skew_angle = search_with_order_iterations(image, min_angle, max_angle, num_iterations)
    
    if debug:
        print(f"Predicted skew angle: {pred_skew_angle}")
    
    # 旋转图像
    deskewed_img = scipy.ndimage.rotate(image, angle=-pred_skew_angle, reshape=False, mode='mirror', order=1)
    
    return deskewed_img, { "skew_angle": pred_skew_angle } 