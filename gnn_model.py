"""
Graph Neural Network model for Link Prediction.
Uses GraphSAGE (SAGEConv) layers from PyTorch Geometric.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SocialGNN(torch.nn.Module):
    """
    Graph Neural Network for Social Link Prediction.
    
    Architecture:
        - Layer 1: SAGEConv (input_features -> hidden_channels)
        - Layer 2: SAGEConv (hidden_channels -> output_channels)
    
    Uses GraphSAGE for inductive learning on graph-structured data.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 64):
        """
        Initialize the SocialGNN model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden channels (default: 64)
            out_channels: Number of output channels/embedding size (default: 64)
        """
        super(SocialGNN, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate node embeddings.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # First GraphSAGE layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        
        return x
    
    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Decode node embeddings to predict edge probabilities using dot product.
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_label_index: Edge indices to predict [2, num_edges_to_predict]
            
        Returns:
            Edge probabilities [num_edges_to_predict]
        """
        # Get embeddings for source and target nodes
        src = z[edge_label_index[0]]  # Source node embeddings
        dst = z[edge_label_index[1]]  # Target node embeddings
        
        # Dot product between source and target embeddings
        # Returns probability score for each edge
        return (src * dst).sum(dim=-1)
    
    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode all possible edges (for inference/visualization).
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            
        Returns:
            Probability matrix [num_nodes, num_nodes]
        """
        # Compute dot product between all pairs of nodes
        prob_matrix = torch.sigmoid(torch.mm(z, z.t()))
        return prob_matrix


if __name__ == "__main__":
    print("🧠 GNN Model Architecture Test")
    print("=" * 50)
    
    # Test configuration
    num_nodes = 100
    in_channels = 16
    hidden_channels = 64
    out_channels = 64
    num_edges = 500
    
    # Create dummy data
    print("\n⏳ Creating dummy test data...")
    x = torch.randn(num_nodes, in_channels)  # Random node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
    
    # Initialize model
    print("⏳ Initializing SocialGNN model...")
    model = SocialGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )
    print("✓ Model initialized!")
    
    # Print model architecture
    print("\n📐 MODEL ARCHITECTURE")
    print("-" * 50)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n⏳ Testing forward pass...")
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output embeddings shape: {z.shape}")
    
    # Test decode
    print("\n⏳ Testing decode method...")
    test_edges = torch.randint(0, num_nodes, (2, 10))  # 10 test edges
    with torch.no_grad():
        scores = model.decode(z, test_edges)
        probs = torch.sigmoid(scores)
    print(f"✓ Test edges shape: {test_edges.shape}")
    print(f"✓ Predicted scores shape: {scores.shape}")
    print(f"✓ Sample probabilities: {probs[:5].tolist()}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
