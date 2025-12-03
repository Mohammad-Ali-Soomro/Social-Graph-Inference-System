"""
Training script for the Social Graph Neural Network.
Handles data preparation, feature engineering, and model training.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import OneHotEncoder

from graph_utils import load_graph
from gnn_model import SocialGNN


# Configuration
HIDDEN_CHANNELS = 64
OUTPUT_CHANNELS = 64
LEARNING_RATE = 0.01
EPOCHS = 100
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def prepare_node_features(G, fit_encoders=True, encoders=None):
    """
    Convert node attributes to numerical features using OneHotEncoding.
    
    Args:
        G: NetworkX graph with node attributes
        fit_encoders: Whether to fit new encoders or use existing ones
        encoders: Pre-fitted encoders (required if fit_encoders=False)
        
    Returns:
        Tuple of (feature_tensor, encoders_dict)
    """
    # Extract node attributes in order
    nodes = sorted(G.nodes())
    
    departments = []
    batches = []
    societies = []
    
    for node in nodes:
        attrs = G.nodes[node]
        departments.append(attrs.get('dept', 'Unknown'))
        batches.append(str(attrs.get('batch', 'Unknown')))  # Convert to string
        societies.append(attrs.get('society', 'Unknown'))
    
    # Reshape for sklearn
    dept_array = np.array(departments).reshape(-1, 1)
    batch_array = np.array(batches).reshape(-1, 1)
    society_array = np.array(societies).reshape(-1, 1)
    
    if fit_encoders:
        # Create and fit encoders
        dept_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        batch_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        society_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        dept_encoded = dept_encoder.fit_transform(dept_array)
        batch_encoded = batch_encoder.fit_transform(batch_array)
        society_encoded = society_encoder.fit_transform(society_array)
        
        encoders = {
            'department': dept_encoder,
            'batch': batch_encoder,
            'society': society_encoder
        }
    else:
        # Use pre-fitted encoders
        dept_encoded = encoders['department'].transform(dept_array)
        batch_encoded = encoders['batch'].transform(batch_array)
        society_encoded = encoders['society'].transform(society_array)
    
    # Concatenate all features
    features = np.concatenate([dept_encoded, batch_encoded, society_encoded], axis=1)
    
    # Convert to tensor
    feature_tensor = torch.tensor(features, dtype=torch.float)
    
    return feature_tensor, encoders


def networkx_to_pyg(G, node_features):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph
        node_features: Node feature tensor
        
    Returns:
        PyTorch Geometric Data object
    """
    # Create node ID mapping (NetworkX node IDs to consecutive integers)
    nodes = sorted(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    
    # Convert edges to tensor
    edges = list(G.edges())
    edge_index = torch.tensor([
        [node_mapping[e[0]], node_mapping[e[1]]] for e in edges
    ], dtype=torch.long).t().contiguous()
    
    # Make edges undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create Data object
    data = Data(x=node_features, edge_index=edge_index)
    data.num_nodes = len(nodes)
    data.node_mapping = node_mapping
    data.reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    return data


def train_epoch(model, optimizer, criterion, train_data):
    """
    Train for one epoch.
    
    Args:
        model: SocialGNN model
        optimizer: Adam optimizer
        criterion: BCEWithLogitsLoss
        train_data: Training data with edge splits
        
    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass - get node embeddings
    z = model(train_data.x, train_data.edge_index)
    
    # Decode edge probabilities for positive and negative edges
    pos_edge_index = train_data.edge_label_index[:, train_data.edge_label == 1]
    neg_edge_index = train_data.edge_label_index[:, train_data.edge_label == 0]
    
    # Get predictions
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Combine predictions and labels
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ], dim=0)
    
    # Calculate loss
    loss = criterion(pred, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: SocialGNN model
        data: Data with edge splits
        
    Returns:
        Tuple of (AUC score, accuracy)
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    model.eval()
    
    # Get node embeddings
    z = model(data.x, data.edge_index)
    
    # Get predictions
    pred = model.decode(z, data.edge_label_index)
    pred_probs = torch.sigmoid(pred).cpu().numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    
    # True labels
    true_labels = data.edge_label.cpu().numpy()
    
    # Calculate metrics
    auc = roc_auc_score(true_labels, pred_probs)
    acc = accuracy_score(true_labels, pred_labels)
    
    return auc, acc


def train_model():
    """Main training function."""
    print("🎓 Social Graph Neural Network Training")
    print("=" * 60)
    
    # Step 1: Load graph
    print("\n📊 Step 1: Loading graph...")
    G = load_graph()
    print(f"   ✓ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Step 2: Prepare node features
    print("\n🔧 Step 2: Preparing node features (OneHotEncoding)...")
    node_features, encoders = prepare_node_features(G)
    print(f"   ✓ Feature shape: {node_features.shape}")
    print(f"   ✓ Features per node: {node_features.shape[1]}")
    
    # Step 3: Convert to PyG Data
    print("\n🔄 Step 3: Converting to PyTorch Geometric format...")
    data = networkx_to_pyg(G, node_features)
    print(f"   ✓ Data object created")
    print(f"   ✓ Edge index shape: {data.edge_index.shape}")
    
    # Step 4: Split edges into train/val/test
    print("\n✂️  Step 4: Splitting edges (Train/Val/Test)...")
    transform = RandomLinkSplit(
        num_val=0.15,
        num_test=0.15,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)
    print(f"   ✓ Train edges: {train_data.edge_label.sum().item():.0f} positive, "
          f"{(train_data.edge_label == 0).sum().item():.0f} negative")
    print(f"   ✓ Val edges: {val_data.edge_label.sum().item():.0f} positive, "
          f"{(val_data.edge_label == 0).sum().item():.0f} negative")
    print(f"   ✓ Test edges: {test_data.edge_label.sum().item():.0f} positive, "
          f"{(test_data.edge_label == 0).sum().item():.0f} negative")
    
    # Step 5: Initialize model
    print("\n🧠 Step 5: Initializing SocialGNN model...")
    in_channels = node_features.shape[1]
    model = SocialGNN(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUTPUT_CHANNELS
    )
    print(f"   ✓ Model: {model.__class__.__name__}")
    print(f"   ✓ Input channels: {in_channels}")
    print(f"   ✓ Hidden channels: {HIDDEN_CHANNELS}")
    print(f"   ✓ Output channels: {OUTPUT_CHANNELS}")
    
    # Step 6: Setup training
    print("\n⚙️  Step 6: Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f"   ✓ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   ✓ Loss: BCEWithLogitsLoss")
    
    # Step 7: Training loop
    print(f"\n🏋️  Step 7: Training for {EPOCHS} epochs...")
    print("-" * 60)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        loss = train_epoch(model, optimizer, criterion, train_data)
        
        # Evaluate
        train_auc, train_acc = evaluate(model, train_data)
        val_auc, val_acc = evaluate(model, val_data)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    
    print("-" * 60)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Step 8: Final evaluation
    print("\n📈 Step 8: Final Evaluation...")
    test_auc, test_acc = evaluate(model, test_data)
    print(f"   ✓ Test AUC: {test_auc:.4f}")
    print(f"   ✓ Test Accuracy: {test_acc:.4f}")
    print(f"   ✓ Best Val AUC: {best_val_auc:.4f}")
    
    # Step 9: Save model and encoders
    print("\n💾 Step 9: Saving model and encoders...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/social_gnn.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'in_channels': in_channels,
        'hidden_channels': HIDDEN_CHANNELS,
        'out_channels': OUTPUT_CHANNELS,
        'test_auc': test_auc,
        'test_acc': test_acc
    }, model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Save encoders
    encoders_path = 'models/encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"   ✓ Encoders saved to: {encoders_path}")
    
    # Save node mapping for inference
    mapping_path = 'models/node_mapping.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'node_mapping': data.node_mapping,
            'reverse_mapping': data.reverse_mapping
        }, f)
    print(f"   ✓ Node mapping saved to: {mapping_path}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Final Test AUC: {test_auc:.4f}")
    print(f"   Final Test Accuracy: {test_acc:.4f}")
    print("=" * 60)
    
    return model, encoders, data


if __name__ == "__main__":
    train_model()
