import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from itertools import combinations_with_replacement

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 多項式特徵轉換
class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self.n_output_features_ = None
        self.powers_ = None
        
    def fit_transform(self, X):
        """生成多項式特徵"""
        n_samples, n_features = X.shape
        
        # 生成所有可能的特徵組合
        combinations = []
        for d in range(0, self.degree + 1):
            combinations.extend(combinations_with_replacement(range(n_features), d))
        
        self.powers_ = np.vstack([np.bincount(c, minlength=n_features) for c in combinations])
        self.n_output_features_ = len(self.powers_)
        
        # 轉換特徵
        X_tensor = torch.FloatTensor(X)
        X_poly = torch.ones((n_samples, self.n_output_features_))
        for i, powers in enumerate(self.powers_):
            if np.any(powers):  # 跳過常數項
                X_poly[:, i] = torch.prod(X_tensor ** torch.FloatTensor(powers), dim=1)
        
        return X_poly
    
    # 轉換新的數據
    def transform(self, X):
        n_samples = X.shape[0]
        X_tensor = torch.FloatTensor(X)
        X_poly = torch.ones((n_samples, self.n_output_features_))
        for i, powers in enumerate(self.powers_):
            if np.any(powers):
                X_poly[:, i] = torch.prod(X_tensor ** torch.FloatTensor(powers), dim=1)
        return X_poly

# RSM 回歸模型，使用線性回歸
class RSMModel(nn.Module):
    def __init__(self, input_size):
        super(RSMModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

# 訓練 RSM 模型
def train_model(points, values, min_r2=0.8, learning_rate=0.01, max_epochs=1000):
    # 確保輸入數據格式正確
    points = np.array(points, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    
    # 如果 values 是單一值，需要重塑它
    if len(values.shape) == 0:
        values = values.reshape(-1)
    
    # 分割訓練集和驗證集
    n_samples = len(points)
    if n_samples < 5:  # 如果樣本太少，就不分割
        X_train, y_train = points, values
        X_val, y_val = points, values
    else:
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        X_train, y_train = points[train_idx], values[train_idx]
        X_val, y_val = points[val_idx], values[val_idx]

    best_model = None
    best_poly = None
    best_degree = None
    best_r2 = float("-inf")

    # 嘗試不同的多項式次數
    for degree in range(2, 5):
        try:
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X_train)
            X_val_poly = poly.transform(X_val)
            
            # 建立模型
            model = RSMModel(X_poly.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # 轉換為 tensor
            X_train_tensor = torch.FloatTensor(X_poly)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val_poly)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # 建立資料載入器
            train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=min(32, len(X_train)), shuffle=True)
            
            # 訓練模型
            model.train()
            for epoch in range(max_epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 評估模型
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor).squeeze().numpy()
                val_preds = np.array(val_preds).reshape(-1)  # 確保是 1D array
                y_val = np.array(y_val).reshape(-1)  # 確保是 1D array
                r2 = r2_score(y_val, val_preds)
                
            print(f"[train_rsm] RSM degree={degree}, R²={r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_poly = poly
                best_degree = degree
                
            if r2 >= min_r2:
                return {
                    "model": best_model,
                    "poly": best_poly,
                    "degree": best_degree,
                    "r2": best_r2,
                    "clusters": None
                }
                
        except Exception as e:
            print(f"Degree {degree} failed: {str(e)}")
            continue

    print("單一 RSM 未達標，返回最佳結果")
    return {
        "model": best_model,
        "poly": best_poly,
        "degree": best_degree,
        "r2": best_r2,
        "clusters": None
    }

def predict(model_data, X):
    """使用訓練好的 RSM 模型進行預測"""
    if model_data["clusters"] is None:
        poly = model_data["poly"]
        model = model_data["model"]
        X_poly = poly.transform(X)
        model.eval()
        with torch.no_grad():
            return model(X_poly).squeeze().numpy()
    
    clusters = model_data["clusters"]
    cluster_models = model_data["model"]
    cluster_polys = model_data["poly"]
    
    total_preds = np.zeros(len(X))
    
    for model, poly in zip(cluster_models, cluster_polys):
        X_poly = poly.transform(X)
        model.eval()
        with torch.no_grad():
            preds = model(X_poly).squeeze().numpy()
            total_preds += preds
    
    return total_preds / len(cluster_models)