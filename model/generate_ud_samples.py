import numpy as np

def generate_uniform_design(n, s):
    """
    生成均勻設計點 (Uniform Design)
    
    參數:
    n (int): 樣本點數
    s (int): 因子數 (變數數量)
    
    返回:
    numpy.ndarray: 生成的均勻設計點 (n x s) 矩陣
    """
    H = np.zeros((n, s))
    for j in range(s):
        for i in range(n):
            H[i, j] = ((i + 1) * (j + 1)) % n
    return H / n

def generate_ud_samples(num_samples, bounds):
    """
    根據上下界生成 Uniform Design (UD) 樣本，並限制數值為 2 位小數
    
    :param num_samples: 生成的樣本數量
    :param bounds: 變數上下界 [[下界, 上界], [下界, 上界], ...]
    :return: 生成的樣本點陣列 (num_samples, num_dimensions) 並限制小數點 2 位
    """
    bounds = np.array(bounds, dtype=np.float64)  # 確保是 NumPy 浮點數陣列
    num_dimensions = bounds.shape[0]  # 取得 X 維度數

    # 生成 UD 設計點
    uniform_design = generate_uniform_design(num_samples, num_dimensions)
    
    # 根據上下界調整樣本範圍
    scaled_samples = bounds[:, 0] + uniform_design * (bounds[:, 1] - bounds[:, 0])
    
    # **🔥 限制數值到 2 位小數**
    scaled_samples = np.round(scaled_samples, 2)

    return scaled_samples