import numpy as np
from scipy.stats import qmc

def generate_lhs_samples(num_samples, bounds):
    """
    使用拉丁超立方體（LHS）生成樣本點

    :param num_samples: 生成的樣本數量
    :param bounds: 變數上下界 [[下界, 上界], [下界, 上界], ...]
    :return: 生成的樣本點陣列 (num_samples, num_dimensions)，數值四捨五入至2位小數
    """
    bounds = np.array(bounds, dtype=np.float64)  # 確保是 NumPy 陣列
    num_dimensions = bounds.shape[0]  # 取得 X 維度數

    # 使用 LHS 產生樣本
    lhs = qmc.LatinHypercube(d=num_dimensions)
    sample = lhs.random(n=num_samples)

    # 調整樣本範圍
    scaled_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    scaled_samples = np.round(scaled_samples, 2)  # 四捨五入到 2 位小數

    return scaled_samples.tolist()  # 轉換為 Python 列表格式，方便 JSON 傳輸
