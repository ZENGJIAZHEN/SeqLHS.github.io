import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def create_latin_hypercube_sampling(block_mapping, n_samples, bounds, existing_points):
    """
    修正的 LHS 採樣函數，支援任意維度
    """
    n_dim = len(bounds)  # 動態獲取維度數
    selected_points = []
    
    # 建立多維度的使用狀態追蹤
    used_indices = [set() for _ in range(n_dim)]
    
    # 根據興趣值排序區塊
    sorted_blocks = sorted(
        block_mapping.items(),
        key=lambda x: x[1]['interest_value'],
        reverse=True
    )
    
    # 為每個樣本選擇點
    for i in range(n_samples):
        # 過濾掉已使用的索引
        remaining_blocks = []
        for indices, block in sorted_blocks:
            if len(indices) != n_dim:  # 確保索引維度正確
                continue
            # 檢查每個維度是否可用
            if not any(indices[d] in used_indices[d] for d in range(n_dim)):
                remaining_blocks.append((indices, block))
        
        if not remaining_blocks:
            print(f"⚠️ 警告：在選擇第 {i+1} 個點時無可用區塊")
            break
        
        # 選擇最優區塊
        best_indices, best_block = remaining_blocks[0]
        
        # 生成點
        point = []
        for dim in range(n_dim):
            dim_range = best_block['ranges'][dim]
            min_val, max_val = dim_range
            # 在區間內隨機生成
            center = (min_val + max_val) / 2
            width = max_val - min_val
            offset = (np.random.random() - 0.5) * 0.6 * width
            value = np.clip(center + offset, min_val, max_val)
            point.append(value)
            
            # 更新使用狀態
            if len(used_indices[dim]) < n_samples:
                used_indices[dim].add(best_indices[dim])
        
        selected_points.append(point)
        print(f"✅ 已選擇點 {point}")
    
    result = np.array(selected_points)
    print(f"Final shape of selected points: {result.shape}")
    return result

def update_search_space(current_points, current_values, current_bounds, n_samples, d_reduction=1):
    """
    更新搜索空間
    """
    n_dim = current_points.shape[1]  # 動態獲取維度數
    
    # 找到目前最佳點
    best_idx = np.argmax(current_values)
    best_point = current_points[best_idx]
    
    # 計算分割數
    n_divisions = max(n_samples + 1, int(np.sqrt(len(current_points))))
    print(f"搜索空間將被分割為 {n_divisions} 份")
    
    # 收縮搜索空間
    shrink_ratio = 0.7
    original_range = current_bounds[:, 1] - current_bounds[:, 0]
    new_range = original_range * shrink_ratio
    
    # 計算新的邊界
    new_bounds = np.zeros((n_dim, 2))
    for i in range(n_dim):
        center = best_point[i]
        half_range = new_range[i] / 2
        new_bounds[i] = [
            max(center - half_range, current_bounds[i, 0]),
            min(center + half_range, current_bounds[i, 1])
        ]
    
    print(f"新的搜索範圍：\n{new_bounds}")
    return new_bounds, n_divisions

def update_search_space(current_points, current_values, current_bounds, n_samples, d_reduction):
    """更新搜索範圍，確保範圍變成原始範圍的 70%"""
    n_dim = current_points.shape[1]
    
    # 找到目前最佳點
    best_idx = np.argmax(current_values)
    best_point = current_points[best_idx]

    # 計算 n_divisions，確保區塊夠多
    n_divisions = max(n_samples + 1, int(np.sqrt(len(current_points))))

    print(f"🔹 計算新搜索範圍，n_divisions = {n_divisions}")

    # **🔹 讓範圍變成原始範圍的 70%**
    shrink_ratio = 0.7  # **新範圍應該是原始範圍的 70%**
    
    original_range = current_bounds[:, 1] - current_bounds[:, 0]
    new_range = original_range * shrink_ratio  # 直接設定新範圍的長度

    new_bounds = np.zeros((n_dim, 2))

    for i in range(n_dim):
        # 讓範圍以最佳點為中心
        new_bounds[i] = [
            best_point[i] - new_range[i] / 2,
            best_point[i] + new_range[i] / 2
        ]

        # **確保範圍不超過原始範圍**
        new_bounds[i, 0] = max(new_bounds[i, 0], current_bounds[i, 0])
        new_bounds[i, 1] = min(new_bounds[i, 1], current_bounds[i, 1])

    print(f"✅ 調整後的新搜索範圍: {new_bounds.tolist()}")

    return new_bounds, n_divisions
