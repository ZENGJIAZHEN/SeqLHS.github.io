import numpy as np
from sklearn.metrics import r2_score
from train_rsm import predict as predict_rsm
from train_mlp import predict_mlp

# 評估模型的準確度，計算 R2 和平均相對誤差
def evaluate_model_accuracy(current_points, current_values, model_data):
    if isinstance(model_data, dict):  # RSM
        predictions = predict_rsm(model_data, current_points)
    else:  # MLP
        predictions = predict_mlp(model_data, current_points)
    
    # 平均相對誤差
    relative_errors = np.abs(predictions - current_values) / (np.abs(current_values) + 1e-10)
    mean_relative_error = np.mean(relative_errors)
    
    # R2
    r2 = r2_score(current_values, predictions)
    
    print(f"Model Evaluation Results:")
    print(f"Mean Relative Error: {mean_relative_error:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return r2 > 0.5

# 計算網格點的興趣值
def calculate_interest_value(current_points, current_values, model_data, bounds, n_divisions, min_r2=0.8):
    n_dim = bounds.shape[0]
    
    # 建立網格
    grids = [
        np.linspace(bounds[i, 0], bounds[i, 1], n_divisions)
        for i in range(n_dim)
    ]
    
    # 建立網格點
    mesh = np.meshgrid(*grids, indexing='ij')
    grid_points = np.stack([m.flatten() for m in mesh], axis=-1)
    
    # 添加調試信息
    print(f"Grid points shape: {grid_points.shape}")
    
    # 評估模型準確度
    if isinstance(model_data, dict):  # RSM
        predictions = predict_rsm(model_data, current_points)
    else:  # MLP
        predictions = predict_mlp(model_data, current_points)
    
    r2 = r2_score(current_values, predictions)
    
    if r2 >= min_r2:
        # 模型準確 - 直接使用預測值
        if isinstance(model_data, dict):
            interest_values = predict_rsm(model_data, grid_points)
        else:
            try:
                interest_values = predict_mlp(model_data, grid_points)
            except Exception as e:
                print(f"Error predicting interest values: {str(e)}")
                # 如果預測失敗，回退到距離計算
                r2 = min_r2 - 0.1  # 強制使用距離計算
    
    if r2 < min_r2:
        # 模型不準確或預測失敗 - 使用實驗趨勢優先的方式
        best_idx = np.argmax(current_values)
        best_point = current_points[best_idx]
        
        # 計算到最佳點的距離
        d_x = np.linalg.norm(grid_points - best_point, axis=1)
        d_min, d_max = np.min(d_x), np.max(d_x)
        
        # 修正距離權重
        D_x = (d_max - d_x) / (d_max - d_min + 1e-10)
        
        # 使用模型預測值（即使不準確也先試試）
        try:
            if isinstance(model_data, dict):
                f_x = predict_rsm(model_data, grid_points)
            else:
                f_x = predict_mlp(model_data, grid_points)
        except:
            # 如果預測失敗，使用簡單的距離權重
            f_x = D_x
            
        f_max = np.max(f_x)
        f_min = np.min(f_x)
        if f_max - f_min < 1e-10:
            normalized_f_x = np.ones_like(f_x)
        else:
            normalized_f_x = (f_x - f_min) / (f_max - f_min + 1e-10)
        
        interest_values = (1 + normalized_f_x) * (1 + D_x)
    
    return interest_values.reshape([n_divisions] * n_dim)

# 建立區塊映射
def create_block_mapping(interest_values, bounds, n_divisions):
    n_dim = bounds.shape[0]
    
    # 建立網格
    grids = [
        np.linspace(bounds[i, 0], bounds[i, 1], n_divisions + 1)
        for i in range(n_dim)
    ]
    
    # 建立區塊映射
    block_mapping = {}
    grid_indices = np.array(np.unravel_index(
        np.arange(n_divisions**n_dim),
        [n_divisions] * n_dim
    )).T
    
    for idx in grid_indices:
        ranges = [
            [grids[dim][i], grids[dim][i + 1]]
            for dim, i in enumerate(idx)
        ]
        block_mapping[tuple(idx)] = {
            'interest_value': float(interest_values[tuple(idx)]),
            'ranges': ranges
        }
    
    return block_mapping

# 生成新的採樣點
def create_latin_hypercube_sampling(block_mapping, n_samples, bounds, existing_points):
    n_dim = len(bounds)
    selected_points = []
    
    # 為每個維度建立使用狀態追蹤
    used_indices = [set() for _ in range(n_dim)]
    
    # 根據興趣值排序區塊
    sorted_blocks = sorted(
        block_mapping.items(),
        key=lambda x: x[1]['interest_value'],
        reverse=True
    )
    
    for i in range(n_samples):
        # 過濾已使用的索引
        remaining_blocks = [
            (indices, block) for indices, block in sorted_blocks
            if not any(idx in used_indices[dim] 
                      for dim, idx in enumerate(indices))
        ]
        
        if not remaining_blocks:
            print(f"Warning: No available blocks after {i} points")
            break
        
        # 選擇最佳區塊
        best_indices, best_block = remaining_blocks[0]
        
        # 生成點
        point = []
        for dim, (min_val, max_val) in enumerate(best_block['ranges']):
            center = (min_val + max_val) / 2
            width = max_val - min_val
            offset = (np.random.random() - 0.5) * 0.6 * width
            value = np.clip(center + offset, min_val, max_val)
            point.append(value)
            
            # 更新使用狀態
            if len(used_indices[dim]) < n_samples:
                used_indices[dim].add(best_indices[dim])
        
        selected_points.append(point)
    
    return np.array(selected_points)

# 縮小搜尋範圍
def update_search_space(current_points, current_values, current_bounds, n_samples, d_reduction=1):
    n_dim = current_points.shape[1]
    
    # 找出目前的最佳點
    best_idx = np.argmax(current_values)
    best_point = current_points[best_idx]

    # 根據當前點數計算新的網格劃分數
    n_divisions = max(n_samples + 1, int(np.sqrt(len(current_points))))
    print(f"Calculated {n_divisions} divisions for new search space")

    # 縮小搜尋範圍
    shrink_ratio = 0.7  # 這裡設定範圍縮小 70%
    original_range = current_bounds[:, 1] - current_bounds[:, 0]
    new_range = original_range * shrink_ratio

    # 初始化新的搜尋邊界
    new_bounds = np.zeros((n_dim, 2))
    for i in range(n_dim):
        # 設定新邊界，使其以「當前最佳點」為中心
        new_bounds[i] = [
            best_point[i] - new_range[i] / 2,
            best_point[i] + new_range[i] / 2
        ]

        # 確保新的邊界不會超出原始範圍
        new_bounds[i, 0] = max(new_bounds[i, 0], current_bounds[i, 0])
        new_bounds[i, 1] = min(new_bounds[i, 1], current_bounds[i, 1])

    print(f"New search bounds: {new_bounds.tolist()}")
    return new_bounds, n_divisions