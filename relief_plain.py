import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class Relief:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights = torch.zeros(X.shape[1])
        self.classes = torch.unique(y)
        
        self._compute_weights()
        
    def _compute_weights(self):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        for i in range(self.X.shape[0]):
            Ri = X_scaled[i]
            yi = self.y[i]
            
            same_class_idx = (self.y == yi).nonzero().squeeze()
            diff_class_idx = (self.y != yi).nonzero().squeeze()
            
            nn_same_class = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X[same_class_idx])
            nn_diff_class = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X[diff_class_idx])
            
            _, idx_same_class = nn_same_class.kneighbors([Ri], self.n_neighbors, return_distance=False)
            _, idx_diff_class = nn_diff_class.kneighbors([Ri], self.n_neighbors, return_distance=False)
            
            hit = X_scaled[same_class_idx[idx_same_class]].mean(axis=1)
            miss = X_scaled[diff_class_idx[idx_diff_class]].mean(axis=1)
            
            self.weights -= torch.abs(Ri - hit).sum(axis=0) / (self.X.shape[0] * self.n_neighbors)
            self.weights += torch.abs(Ri - miss).sum(axis=0) / (self.X.shape[0] * self.n_neighbors)
        
    def transform(self, X):
        return X * self.weights

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)
    return X, y

if __name__ == "__main__":
    csv_file = 'dataset/wine.csv'  # 替换为你的CSV文件路径
    X, y = load_data(csv_file)
    
    relief = Relief(n_neighbors=10)
    relief.fit(X, y)
    
    print("Feature Weights:", relief.weights)
    
    X_transformed = relief.transform(X)
    print("Transformed Data:", X_transformed)
