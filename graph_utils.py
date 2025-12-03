"""
Graph utilities for loading and analyzing the university social graph.
Uses NetworkX for graph operations and analysis.
"""

import os
import pandas as pd
import networkx as nx


def load_graph(nodes_path: str = 'data/nodes.csv', edges_path: str = 'data/edges.csv') -> nx.Graph:
    """
    Load the social graph from CSV files.
    
    Args:
        nodes_path: Path to the nodes CSV file
        edges_path: Path to the edges CSV file
        
    Returns:
        NetworkX Graph object with node attributes (dept, batch, society)
    """
    # Read CSV files
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(
            row['student_id'],
            name=row['name'],
            dept=row['department'],
            batch=row['batch'],
            society=row['society']
        )
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    return G


def get_graph_stats(G: nx.Graph) -> dict:
    """
    Calculate basic graph statistics.
    
    Args:
        G: NetworkX Graph object
        
    Returns:
        Dictionary containing:
        - total_nodes: Number of nodes
        - total_edges: Number of edges
        - average_degree: Average node degree
        - density: Graph density
    """
    # Total nodes and edges
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    average_degree = sum(degrees) / total_nodes if total_nodes > 0 else 0
    
    # Graph density
    density = nx.density(G)
    
    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'average_degree': round(average_degree, 2),
        'density': round(density, 4)
    }


def get_centrality_measures(G: nx.Graph, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate degree centrality and return top N students.
    
    Args:
        G: NetworkX Graph object
        top_n: Number of top students to return
        
    Returns:
        DataFrame with top students by degree centrality
    """
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Create list of students with their centrality and attributes
    students_data = []
    for node_id, centrality in degree_centrality.items():
        node_attrs = G.nodes[node_id]
        students_data.append({
            'student_id': node_id,
            'name': node_attrs.get('name', 'Unknown'),
            'department': node_attrs.get('dept', 'Unknown'),
            'batch': node_attrs.get('batch', 'Unknown'),
            'society': node_attrs.get('society', 'Unknown'),
            'degree_centrality': round(centrality, 4),
            'connections': G.degree(node_id)
        })
    
    # Create DataFrame and sort by degree centrality
    df = pd.DataFrame(students_data)
    df = df.sort_values('degree_centrality', ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # Start index from 1
    
    return df


def get_node_info(G: nx.Graph, student_id: int) -> dict:
    """
    Get information about a specific student node.
    
    Args:
        G: NetworkX Graph object
        student_id: ID of the student
        
    Returns:
        Dictionary with student information and connections
    """
    if student_id not in G.nodes:
        return None
    
    node_attrs = G.nodes[student_id]
    neighbors = list(G.neighbors(student_id))
    
    return {
        'student_id': student_id,
        'name': node_attrs.get('name', 'Unknown'),
        'department': node_attrs.get('dept', 'Unknown'),
        'batch': node_attrs.get('batch', 'Unknown'),
        'society': node_attrs.get('society', 'Unknown'),
        'num_connections': len(neighbors),
        'connected_to': neighbors[:10]  # First 10 connections
    }


if __name__ == "__main__":
    print("🔗 Graph Utilities Test")
    print("=" * 50)
    
    # Load the graph
    print("\n⏳ Loading graph from CSV files...")
    G = load_graph()
    print("✓ Graph loaded successfully!")
    
    # Get and print graph statistics
    print("\n📊 GRAPH STATISTICS")
    print("-" * 50)
    stats = get_graph_stats(G)
    print(f"   • Total Nodes (Students): {stats['total_nodes']}")
    print(f"   • Total Edges (Connections): {stats['total_edges']}")
    print(f"   • Average Degree: {stats['average_degree']}")
    print(f"   • Graph Density: {stats['density']}")
    
    # Get and print centrality measures
    print("\n🏆 TOP 5 STUDENTS BY DEGREE CENTRALITY (Most Popular)")
    print("-" * 50)
    centrality_df = get_centrality_measures(G, top_n=5)
    print(centrality_df.to_string())
    
    # Test node info for top student
    top_student_id = centrality_df.iloc[0]['student_id']
    print(f"\n👤 SAMPLE NODE INFO (Student ID: {top_student_id})")
    print("-" * 50)
    node_info = get_node_info(G, top_student_id)
    if node_info:
        print(f"   • Name: {node_info['name']}")
        print(f"   • Department: {node_info['department']}")
        print(f"   • Batch: {node_info['batch']}")
        print(f"   • Society: {node_info['society']}")
        print(f"   • Total Connections: {node_info['num_connections']}")
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")
