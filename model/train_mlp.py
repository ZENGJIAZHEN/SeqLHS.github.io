import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score

# 自訂數據集類別
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# MLP 模型定義
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layers, activation='relu'):
        super(MLPRegressor, self).__init__()
        
        # 定義網路層大小 (輸入層 + 隱藏層 + 輸出層)
        layer_sizes = [input_size] + list(hidden_layers) + [1]
        
        # 建立神經網路層
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:  # 最後一層不加激活函數
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'logistic':
                    layers.append(nn.Sigmoid())
        # 將層組合成一個網路
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_mlp(X, y, hidden_layers=3, min_r2=0.8, activation='relu', 
              max_iter=2000, learning_rate=0.001 ,random_state=42, batch_size=32):
    # 設定隨機種子
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    else:
        torch.seed()
        np.random.seed()
    
    if isinstance(hidden_layers, int):
        hidden_layers = (hidden_layers,)
    
    # 建立數據集
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化 MLP 模型
    input_size = X.shape[1]
    model = MLPRegressor(input_size, hidden_layers, activation)
    
    # 設定損失函數與優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練模型
    model.train()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(max_iter):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # 前向傳播
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        
        # 提早停止 (Early Stopping)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 計算 R2 分數
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X)).squeeze().numpy()
        r2 = r2_score(y, predictions)
    
    # 判斷是否需要調整模型參數
    needs_param_change = r2 < min_r2
    
    print(f"[train_mlp] hidden={hidden_layers}, activation={activation}, "
          f"max_iter={max_iter}, learning_rate={learning_rate}, batch_size={batch_size}, random_state={random_state}, "
          f"R2={r2:.4f}, loss={best_loss:.4f}")
    
    return model, r2, best_loss, needs_param_change

# 使用訓練好的 MLP 模型進行預測
def predict_mlp(model, X):
    try:
        X = np.array(X, dtype=np.float32)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) > 2:
            # 如果是高維數組，重塑為 2D
            n_features = X.shape[-1]
            X = X.reshape(-1, n_features)
        
        X_tensor = torch.FloatTensor(X)
        
        # 進行預測
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).squeeze().numpy()
            

            if np.isscalar(predictions):
                predictions = np.array([predictions])
                
            return predictions
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Input shape: {X.shape}")
        raise
