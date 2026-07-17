import logging
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import Tuple, List

logger = logging.getLogger("GNN-Spatial-Forecaster")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SpatialGNNForecaster(torch.nn.Module):
    """
    Graph Neural Network model capturing spatial correlations between 
    1km x 1km grid cells for air pollution diffusion forecasting.
    """
    def __init__(self, num_node_features: int, hidden_channels: int = 32):
        super(SpatialGNNForecaster, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1) # Predicts PM2.5 delta

    def forward(self, x, edge_index):
        # 1st Graph Convolutional Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 2nd Graph Convolutional Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output prediction mapping
        out = self.linear(x)
        return out

class GNNTrainer:
    """Handles data preparation, training, and inference for the GNN model."""
    def __init__(self, num_features: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatialGNNForecaster(num_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        logger.info(f"Initialized GNN on device: {self.device}")

    def create_graph_data(self, df: pd.DataFrame, edge_list: List[Tuple[int, int]]) -> Data:
        """
        Converts tabular grid data into a PyTorch Geometric Data object.
        Edges represent adjacent grid cells (diffusion pathways).
        """
        # Node features (excluding IDs and targets)
        feature_cols = [c for c in df.columns if c not in ['grid_id', 'timestamp', 'pm25_target']]
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        
        # Target variable
        y = torch.tensor(df['pm25_target'].values, dtype=torch.float).view(-1, 1) if 'pm25_target' in df else None
        
        # Edge index [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data.to(self.device)

    def train_epoch(self, data: Data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, data: Data) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index)
        return pred.cpu().numpy()

if __name__ == "__main__":
    logger.info("GNN Forecaster Smoke Test")
    # 4 grid cells, linear connections 0-1-2-3
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    df_mock = pd.DataFrame({
        'feat1': np.random.rand(4),
        'feat2': np.random.rand(4),
        'pm25_target': np.random.rand(4) * 100
    })
    
    trainer = GNNTrainer(num_features=2)
    graph_data = trainer.create_graph_data(df_mock, edges)
    
    for epoch in range(10):
        loss = trainer.train_epoch(graph_data)
        logger.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
