"""
Inference module for friend recommendations.
Uses the trained GNN model to predict potential connections.
"""

import pickle
import random
import torch
import pandas as pd

from graph_utils import load_graph, get_node_info
from gnn_model import SocialGNN
from train_model import prepare_node_features, networkx_to_pyg


def load_trained_model(model_path='models/social_gnn.pth'):
    """
    Load the trained SocialGNN model.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(model_path, weights_only=False)
    
    model = SocialGNN(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels'],
        out_channels=checkpoint['out_channels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_encoders(encoders_path='models/encoders.pkl'):
    """
    Load the saved feature encoders.
    
    Args:
        encoders_path: Path to the saved encoders
        
    Returns:
        Dictionary of encoders
    """
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    return encoders


def load_node_mapping(mapping_path='models/node_mapping.pkl'):
    """
    Load the node ID mappings.
    
    Args:
        mapping_path: Path to the saved mappings
        
    Returns:
        Dictionary with node_mapping and reverse_mapping
    """
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    return mappings


def get_node_embeddings(model, data):
    """
    Get node embeddings from the trained model.
    
    Args:
        model: Trained SocialGNN model
        data: PyTorch Geometric Data object
        
    Returns:
        Node embedding tensor
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    return embeddings


def recommend_friends(student_id, model, data, graph, node_mapping, top_k=5):
    """
    Recommend potential friends for a specific student.
    
    Args:
        student_id: The student ID to get recommendations for
        model: Trained SocialGNN model
        data: PyTorch Geometric Data object
        graph: NetworkX graph
        node_mapping: Dictionary mapping student IDs to node indices
        top_k: Number of recommendations to return
        
    Returns:
        List of dictionaries with recommended friends and scores
    """
    # Check if student exists
    if student_id not in node_mapping:
        raise ValueError(f"Student ID {student_id} not found in the graph")
    
    # Get node index for the target student
    target_idx = node_mapping[student_id]
    
    # Get all node embeddings
    embeddings = get_node_embeddings(model, data)
    
    # Get the target student's embedding
    target_embedding = embeddings[target_idx]
    
    # Calculate similarity scores (dot product) with all other nodes
    similarity_scores = torch.mm(
        target_embedding.unsqueeze(0), 
        embeddings.t()
    ).squeeze()
    
    # Convert to probabilities using sigmoid
    probabilities = torch.sigmoid(similarity_scores)
    
    # Get current friends (existing connections)
    current_friends = set(graph.neighbors(student_id))
    current_friends.add(student_id)  # Exclude self
    
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    # Filter out existing friends and self, then get top k
    recommendations = []
    
    # Get all scores with their indices
    scores_with_idx = [(idx, prob.item()) for idx, prob in enumerate(probabilities)]
    
    # Sort by probability (descending)
    scores_with_idx.sort(key=lambda x: x[1], reverse=True)
    
    # Filter and get top k
    for idx, score in scores_with_idx:
        original_id = reverse_mapping[idx]
        
        # Skip if already friends or self
        if original_id in current_friends:
            continue
        
        # Get student info
        node_attrs = graph.nodes[original_id]
        
        recommendations.append({
            'student_id': original_id,
            'name': node_attrs.get('name', 'Unknown'),
            'department': node_attrs.get('dept', 'Unknown'),
            'batch': node_attrs.get('batch', 'Unknown'),
            'society': node_attrs.get('society', 'Unknown'),
            'similarity_score': round(score, 4),
            'confidence': f"{score * 100:.1f}%"
        })
        
        if len(recommendations) >= top_k:
            break
    
    return recommendations


def get_connection_probability(student_id_1, student_id_2, model, data, node_mapping):
    """
    Get the probability of connection between two specific students.
    
    Args:
        student_id_1: First student ID
        student_id_2: Second student ID
        model: Trained SocialGNN model
        data: PyTorch Geometric Data object
        node_mapping: Dictionary mapping student IDs to node indices
        
    Returns:
        Connection probability (0-1)
    """
    # Get node indices
    idx_1 = node_mapping[student_id_1]
    idx_2 = node_mapping[student_id_2]
    
    # Get embeddings
    embeddings = get_node_embeddings(model, data)
    
    # Create edge index for the pair
    edge_index = torch.tensor([[idx_1], [idx_2]], dtype=torch.long)
    
    # Get prediction
    with torch.no_grad():
        score = model.decode(embeddings, edge_index)
        probability = torch.sigmoid(score).item()
    
    return probability


def print_recommendations(student_id, recommendations, graph):
    """Pretty print the recommendations."""
    student_info = get_node_info(graph, student_id)
    
    print("\n" + "=" * 70)
    print("🎓 FRIEND RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n👤 Target Student:")
    print(f"   ID: {student_id}")
    print(f"   Name: {student_info['name']}")
    print(f"   Department: {student_info['department']}")
    print(f"   Batch: {student_info['batch']}")
    print(f"   Society: {student_info['society']}")
    print(f"   Current Friends: {student_info['num_connections']}")
    
    print(f"\n🤝 Top {len(recommendations)} Recommended Friends:")
    print("-" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec['name']} (ID: {rec['student_id']})")
        print(f"      Department: {rec['department']} | Batch: {rec['batch']} | Society: {rec['society']}")
        print(f"      Similarity Score: {rec['similarity_score']:.4f} ({rec['confidence']})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("🔮 Social Graph Inference System")
    print("=" * 50)
    
    # Step 1: Load the graph
    print("\n⏳ Loading graph...")
    graph = load_graph()
    print(f"   ✓ Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Step 2: Load encoders and prepare features
    print("\n⏳ Loading encoders and preparing features...")
    encoders = load_encoders()
    node_features, _ = prepare_node_features(graph, fit_encoders=False, encoders=encoders)
    print(f"   ✓ Features prepared: {node_features.shape}")
    
    # Step 3: Convert to PyG Data
    print("\n⏳ Converting to PyG format...")
    data = networkx_to_pyg(graph, node_features)
    node_mapping = data.node_mapping
    print(f"   ✓ Data object created")
    
    # Step 4: Load trained model
    print("\n⏳ Loading trained model...")
    model, checkpoint = load_trained_model()
    print(f"   ✓ Model loaded (Test AUC: {checkpoint['test_auc']:.4f})")
    
    # Step 5: Get recommendations for a random student
    print("\n⏳ Generating recommendations...")
    
    # Select a random student
    all_student_ids = list(graph.nodes())
    random_student_id = random.choice(all_student_ids)
    
    # Get recommendations
    recommendations = recommend_friends(
        student_id=random_student_id,
        model=model,
        data=data,
        graph=graph,
        node_mapping=node_mapping,
        top_k=5
    )
    
    # Print recommendations
    print_recommendations(random_student_id, recommendations, graph)
    
    # Bonus: Check probability between two specific students
    print("\n📊 BONUS: Connection Probability Check")
    print("-" * 50)
    
    # Pick two random non-connected students
    student_1 = random_student_id
    student_2 = recommendations[0]['student_id'] if recommendations else random.choice(all_student_ids)
    
    prob = get_connection_probability(student_1, student_2, model, data, node_mapping)
    print(f"   Probability that Student {student_1} and Student {student_2}")
    print(f"   should be connected: {prob:.4f} ({prob*100:.1f}%)")
    print("=" * 50)
