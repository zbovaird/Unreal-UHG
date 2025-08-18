# ===================================================================
# GOOGLE COLAB CODE FOR UHG IDS DATA EXPORT TO UNREAL ENGINE 5
# ===================================================================
# Copy and paste this code into Google Colab cells

# Cell 1: Install packages
"""
!pip -q install --upgrade pip
import torch
pt = torch.__version__.split('+')[0]
cuda = torch.version.cuda
if torch.cuda.is_available() and cuda:
  idx = f"https://data.pyg.org/whl/torch-{pt}+cu{cuda.replace('.','')}.html"
else:
  idx = f"https://data.pyg.org/whl/torch-{pt}+cpu.html"

!pip -q install torch_scatter torch_sparse torch_cluster torch_spline_conv -f {idx}
!pip -q install torch_geometric scikit-learn scipy pandas tqdm
"""

# Cell 2: Imports and setup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
import json
import os
import time
from typing import Tuple, Dict, Any

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

# File paths (update these to match your setup)
FILE_PATH = '/content/drive/MyDrive/CIC_data.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/uhg_ids_model.pth'
RESULTS_PATH = '/content/drive/MyDrive/uhg_ids_results'
EXPORT_PATH = '/content/drive/MyDrive/UE5_Export'
os.makedirs(EXPORT_PATH, exist_ok=True)
"""

# Cell 3: UHG primitive functions
"""
def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x[..., :-1] * y[..., :-1]).sum(dim=-1) - x[..., -1] * y[..., -1]

def projective_normalize(points: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    spatial = points[..., :-1]
    time_like = torch.sqrt(torch.clamp(1.0 + (spatial * spatial).sum(dim=-1, keepdim=True), min=eps))
    return torch.cat([spatial, time_like], dim=-1)

def uhg_quadrance_vectorized(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    xx = minkowski_dot(x, x)
    yy = minkowski_dot(y, y)
    xy = minkowski_dot(x, y)
    denom = torch.clamp(xx * yy, min=eps)
    q = 1.0 - (xy * xy) / denom
    return torch.clamp(q, 0.0, 1.0)
"""

# Cell 4: UHG model classes
"""
class UHGMessagePassing(MessagePassing):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features
        self.weight_msg = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_node = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_msg)
        nn.init.xavier_uniform_(self.weight_node)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        features = x[:, :-1]
        z = x[:, -1:]
        transformed_features = features @ self.weight_node
        out = self.propagate(edge_index, x=x, size=None)
        out = out + transformed_features
        out_full = torch.cat([out, z], dim=1)
        out_full = projective_normalize(out_full)
        return out_full

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-uhg_quadrance_vectorized(x_i, x_j))
        messages = (x_j[:, :-1]) @ self.weight_msg
        return messages * weights.view(-1, 1)

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        numerator = scatter_add(inputs, index, dim=0)
        weights_sum = scatter_add(torch.ones_like(inputs), index, dim=0)
        return numerator / torch.clamp(weights_sum, min=1e-6)

class UHGGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        actual_in = in_channels - 1
        self.layers.append(UHGMessagePassing(actual_in, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(UHGMessagePassing(hidden_channels, hidden_channels))
        self.layers.append(UHGMessagePassing(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index)
            spatial = F.relu(h[:, :-1])
            h = torch.cat([spatial, h[:, -1:]], dim=1)
            h = self.dropout(h)
        h = self.layers[-1](h, edge_index)
        return h[:, :-1]
"""

# Cell 5: Load and analyze dataset
"""
print("=== LOADING CIC DATASET ===")
data = pd.read_csv(FILE_PATH, low_memory=False)
data.columns = data.columns.str.strip()
data['Label'] = data['Label'].str.strip()

print(f"Full dataset shape: {data.shape}")
print(f"Dataset columns: {list(data.columns)}")

# Show label distribution
unique_labels = data['Label'].unique()
print(f"\\nUnique labels in dataset: {unique_labels}")
label_counts = data['Label'].value_counts()
print("\\nLabel distribution:")
print(label_counts)

# Create smaller sample for UE5 (5-10% as discussed)
sample_fraction = 0.05  # 5% sample
data_sample = data.sample(frac=sample_fraction, random_state=42)
print(f"\\nSample for UE5 ({sample_fraction*100}%): {data_sample.shape}")

# Show sample label distribution
sample_label_counts = data_sample['Label'].value_counts()
print("\\nSample label distribution:")
print(sample_label_counts)
"""

# Cell 6: Data preprocessing
"""
print("\\n=== PREPROCESSING DATA ===")

# Convert to numeric and handle missing values
data_numeric = data_sample.apply(pd.to_numeric, errors='coerce')
data_filled = data_numeric.fillna(data_numeric.mean())
data_filled = data_filled.replace([np.inf, -np.inf], np.nan)
data_filled = data_filled.fillna(data_filled.max())
if data_filled.isnull().values.any():
    data_filled = data_filled.fillna(0)

labels = data_sample['Label']
features = data_filled.drop(columns=['Label'])

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
labels_numeric = labels.map(label_mapping).values

# Convert to tensors
node_features = torch.tensor(features_scaled, dtype=torch.float32)
labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)

print(f"Processed feature shape: {node_features.shape}")
print(f"Number of classes: {len(label_mapping)}")
print(f"Label mapping: {label_mapping}")
"""

# Cell 7: Create graph structure
"""
print("\\n=== CREATING GRAPH STRUCTURE ===")

k = 2  # Same as your model
features_np = node_features.cpu().numpy()

# Build KNN graph
knn_graph = kneighbors_graph(
    features_np,
    k,
    mode='connectivity',
    include_self=False,
    n_jobs=-1
)
knn_graph_coo = coo_matrix(knn_graph)
edge_index = torch.from_numpy(
    np.vstack((knn_graph_coo.row, knn_graph_coo.col))
).long()

print(f"Graph created with {node_features.size(0)} nodes and {edge_index.size(1)} edges")
print(f"Edge index shape: {edge_index.shape}")

# Add homogeneous coordinate for UHG
node_features_uhg = torch.cat([
    node_features,
    torch.ones(node_features.size(0), 1)
], dim=1)
node_features_uhg = projective_normalize(node_features_uhg)

print(f"UHG feature shape: {node_features_uhg.shape}")
"""

# Cell 8: Load model and get predictions
"""
print("\\n=== LOADING TRAINED MODEL ===")

# Initialize model (same architecture as training)
in_channels = node_features_uhg.size(1)
hidden_channels = 64
out_channels = len(label_mapping)
num_layers = 2

model = UHGGraphSAGE(in_channels, hidden_channels, out_channels, num_layers).to(device)

# Load trained weights if available
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading trained model from: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        node_features_device = node_features_uhg.to(device)
        edge_index_device = edge_index.to(device)
        logits = model(node_features_device, edge_index_device)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        confidence_scores = probabilities.max(dim=1)[0]
    
    # Convert back to CPU for export
    predictions = predictions.cpu().numpy()
    confidence_scores = confidence_scores.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    print(f"Model predictions generated for {len(predictions)} nodes")
    
    # Create reverse label mapping
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    predicted_labels = [idx_to_label[pred] for pred in predictions]
    
    print("\\nPrediction distribution:")
    pred_counts = pd.Series(predicted_labels).value_counts()
    print(pred_counts)
    
else:
    print(f"No trained model found at {MODEL_SAVE_PATH}")
    print("Using ground truth labels for visualization")
    predictions = labels_numeric
    predicted_labels = labels.tolist()
    confidence_scores = np.ones(len(predictions))  # Full confidence for ground truth
    probabilities = np.eye(len(label_mapping))[predictions]  # One-hot for ground truth
"""

# Cell 9: Create network topology
"""
print("\\n=== CREATING NETWORK TOPOLOGY FOR UE5 ===")

# Create nodes with all necessary data
nodes_list = []
for i in range(len(node_features)):
    # Determine attack type and color
    attack_label = predicted_labels[i]
    is_benign = 'BENIGN' in attack_label.upper() or 'NORMAL' in attack_label.upper()
    
    # Color coding for UE5
    if is_benign:
        color = "green"
        threat_level = "benign"
    elif "DOS" in attack_label.upper() or "DDOS" in attack_label.upper():
        color = "red"
        threat_level = "high"
    else:
        color = "orange"
        threat_level = "medium"
    
    node_data = {
        "id": f"node_{i}",
        "label": f"Device_{i}",
        "attack_type": attack_label,
        "predicted_class": int(predictions[i]),
        "confidence": float(confidence_scores[i]),
        "threat_level": threat_level,
        "color": color,
        "is_benign": is_benign,
        "position": {
            "x": float(node_features[i, 0]) * 100,  # Scale for UE5
            "y": float(node_features[i, 1]) * 100,
            "z": float(node_features[i, 2]) * 100 if node_features.size(1) > 2 else 0.0
        },
        "features": node_features[i].tolist()  # Original features
    }
    nodes_list.append(node_data)

print(f"Created {len(nodes_list)} nodes")

# Create edges
edges_list = []
edge_sources = edge_index[0].numpy()
edge_targets = edge_index[1].numpy()

for i, (src, tgt) in enumerate(zip(edge_sources, edge_targets)):
    # Determine edge type based on connected nodes
    src_benign = nodes_list[src]["is_benign"]
    tgt_benign = nodes_list[tgt]["is_benign"]
    
    if src_benign and tgt_benign:
        edge_type = "normal"
        edge_color = "blue"
    elif not src_benign or not tgt_benign:
        edge_type = "suspicious"
        edge_color = "red"
    else:
        edge_type = "unknown"
        edge_color = "gray"
    
    edge_data = {
        "id": f"edge_{i}",
        "source": f"node_{src}",
        "target": f"node_{tgt}",
        "source_idx": int(src),
        "target_idx": int(tgt),
        "edge_type": edge_type,
        "color": edge_color,
        "weight": 1.0
    }
    edges_list.append(edge_data)

print(f"Created {len(edges_list)} edges")
"""

# Cell 10: Create export files
"""
print("\\n=== CREATING EXPORT FILES ===")

# 1. Main network topology
topology_data = {
    "metadata": {
        "dataset_name": "CIC IDS Dataset",
        "model_type": "UHG GraphSAGE",
        "sample_size": len(data_sample),
        "sample_percentage": sample_fraction * 100,
        "total_nodes": len(nodes_list),
        "total_edges": len(edges_list),
        "num_classes": len(label_mapping),
        "features_dim": int(node_features.size(1)),
        "k_neighbors": k,
        "has_model_predictions": os.path.exists(MODEL_SAVE_PATH)
    },
    "label_mapping": label_mapping,
    "attack_types": {
        "benign": [label for label in unique_labels if 'BENIGN' in label.upper() or 'NORMAL' in label.upper()],
        "dos_attacks": [label for label in unique_labels if 'DOS' in label.upper() or 'DDOS' in label.upper()],
        "other_attacks": [label for label in unique_labels if not ('BENIGN' in label.upper() or 'NORMAL' in label.upper() or 'DOS' in label.upper())]
    },
    "nodes": nodes_list,
    "edges": edges_list
}

# Save main topology
topology_path = os.path.join(EXPORT_PATH, 'network_topology.json')
with open(topology_path, 'w') as f:
    json.dump(topology_data, f, indent=2)
print(f"✓ Saved: {topology_path}")

# 2. Statistics for UE5 configuration
stats_data = {
    "network_stats": {
        "total_nodes": len(nodes_list),
        "total_edges": len(edges_list),
        "avg_degree": len(edges_list) * 2 / len(nodes_list),
        "attack_distribution": dict(pd.Series(predicted_labels).value_counts())
    },
    "visualization_hints": {
        "node_colors": {
            "benign": "green",
            "dos_attack": "red",
            "other_attack": "orange"
        },
        "edge_colors": {
            "normal": "blue",
            "suspicious": "red",
            "unknown": "gray"
        },
        "recommended_layout": {
            "node_scale": 100,
            "edge_thickness_base": 1.0,
            "camera_distance": 500
        }
    }
}

stats_path = os.path.join(EXPORT_PATH, 'network_stats.json')
with open(stats_path, 'w') as f:
    json.dump(stats_data, f, indent=2)
print(f"✓ Saved: {stats_path}")

# 3. Raw CSV for backup
csv_path = os.path.join(EXPORT_PATH, 'network_data.csv')
export_df = pd.DataFrame({
    'node_id': [f"node_{i}" for i in range(len(nodes_list))],
    'attack_type': predicted_labels,
    'predicted_class': predictions,
    'confidence': confidence_scores,
    'is_benign': [node['is_benign'] for node in nodes_list],
    'threat_level': [node['threat_level'] for node in nodes_list],
    'pos_x': [node['position']['x'] for node in nodes_list],
    'pos_y': [node['position']['y'] for node in nodes_list],
    'pos_z': [node['position']['z'] for node in nodes_list]
})
export_df.to_csv(csv_path, index=False)
print(f"✓ Saved: {csv_path}")
"""

# Cell 11: Display summary
"""
print("\\n" + "="*50)
print("UE5 EXPORT COMPLETE")
print("="*50)
print(f"📊 Dataset: {len(data_sample)} samples ({sample_fraction*100}% of full dataset)")
print(f"🏠 Nodes: {len(nodes_list)} network devices")
print(f"🔗 Edges: {len(edges_list)} connections")
print(f"🎯 Attack Types: {len(unique_labels)} different categories")
print(f"🤖 Model: {'Loaded and predictions generated' if os.path.exists(MODEL_SAVE_PATH) else 'Using ground truth labels'}")

print("\\n📁 Files created:")
print("• network_topology.json - Main network structure for UE5")
print("• network_stats.json - Statistics and visualization hints")  
print("• network_data.csv - Raw data backup")

print("\\n🎨 Attack Type Distribution:")
attack_dist = pd.Series(predicted_labels).value_counts()
for attack, count in attack_dist.items():
    percentage = (count / len(predicted_labels)) * 100
    print(f"• {attack}: {count} ({percentage:.1f}%)")

print("\\n⬇️ Download commands:")
print("from google.colab import files")
print("files.download('/content/drive/MyDrive/UE5_Export/network_topology.json')")
print("files.download('/content/drive/MyDrive/UE5_Export/network_stats.json')")
print("files.download('/content/drive/MyDrive/UE5_Export/network_data.csv')")

print("\\n🚀 Ready for GitHub upload and UE5 import!")
"""

# Cell 12: Download files
"""
from google.colab import files

print("Downloading files...")
files.download('/content/drive/MyDrive/UE5_Export/network_topology.json')
files.download('/content/drive/MyDrive/UE5_Export/network_stats.json')
files.download('/content/drive/MyDrive/UE5_Export/network_data.csv')
print("All files downloaded!")
"""
