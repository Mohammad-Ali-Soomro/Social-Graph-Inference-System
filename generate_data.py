"""
Generate synthetic university social graph data.
Creates nodes (students) and edges (relationships) CSVs.
Produces a realistic, sparse social graph with Average Degree 8-15 and Density < 0.05.
"""

import os
import random
from itertools import combinations
import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker and set seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
NUM_STUDENTS = 500
DEPARTMENTS = ['CS', 'EE', 'BBA', 'Psychology']
BATCHES = [2021, 2022, 2023, 2024]
SOCIETIES = ['Debating', 'ACM', 'Music', 'Sports', 'None']

# Connection probabilities (Strict - for sparse graph)
BASE_PROBABILITY = 0.0          # No connection by default
SAME_SOCIETY_PROB = 0.08        # 8% if same society (excluding 'None')
SAME_BATCH_DEPT_PROB = 0.05     # 5% if same batch AND department
RANDOM_NOISE_PROB = 0.002       # 0.2% random baseline noise


def generate_students(num_students: int) -> pd.DataFrame:
    """
    Generate synthetic student data using Faker.
    
    Args:
        num_students: Number of students to generate
        
    Returns:
        DataFrame with student information
    """
    students = []
    
    for student_id in range(1, num_students + 1):
        student = {
            'student_id': student_id,
            'name': fake.name(),
            'department': random.choice(DEPARTMENTS),
            'batch': random.choice(BATCHES),
            'society': random.choice(SOCIETIES)
        }
        students.append(student)
    
    return pd.DataFrame(students)


def generate_edges(students_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate edges (relationships) between students based on strict probabilities.
    
    Probability Rules:
    - Base: 0.0 (no connection by default)
    - Same society (excluding 'None'): +0.08 (8%)
    - Same batch AND department: +0.05 (5%)
    - Random noise: +0.002 (0.2%)
    
    Args:
        students_df: DataFrame containing student information
        
    Returns:
        DataFrame with edge connections (source, target)
    """
    edges = []
    
    # Convert DataFrame to dictionary for faster lookup
    students_dict = students_df.set_index('student_id').to_dict('index')
    student_ids = list(students_dict.keys())
    
    # Iterate through all possible pairs using combinations
    for student_a_id, student_b_id in combinations(student_ids, 2):
        student_a = students_dict[student_a_id]
        student_b = students_dict[student_b_id]
        
        # Calculate connection probability
        probability = BASE_PROBABILITY
        
        # Rule 1: Same society (excluding 'None') -> +8%
        if (student_a['society'] == student_b['society'] and 
            student_a['society'] != 'None'):
            probability += SAME_SOCIETY_PROB
        
        # Rule 2: Same batch AND department -> +5%
        if (student_a['batch'] == student_b['batch'] and 
            student_a['department'] == student_b['department']):
            probability += SAME_BATCH_DEPT_PROB
        
        # Rule 3: Random noise -> +0.2%
        probability += RANDOM_NOISE_PROB
        
        # Determine if edge exists based on probability
        if random.random() < probability:
            edges.append({
                'source': student_a_id,
                'target': student_b_id
            })
    
    return pd.DataFrame(edges)


def calculate_graph_stats(num_nodes: int, num_edges: int) -> dict:
    """
    Calculate graph statistics.
    
    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        
    Returns:
        Dictionary with average degree and density
    """
    # Average degree = 2 * edges / nodes (each edge contributes to 2 node degrees)
    average_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    
    # Density = 2 * edges / (nodes * (nodes - 1)) for undirected graph
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0
    
    return {
        'average_degree': round(average_degree, 2),
        'density': round(density, 4)
    }


def save_data(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, data_dir: str = 'data'):
    """
    Save nodes and edges DataFrames to CSV files.
    
    Args:
        nodes_df: DataFrame with student nodes
        edges_df: DataFrame with relationship edges
        data_dir: Directory to save files
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save files
    nodes_path = os.path.join(data_dir, 'nodes.csv')
    edges_path = os.path.join(data_dir, 'edges.csv')
    
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    
    print(f"✓ Saved nodes to: {nodes_path}")
    print(f"✓ Saved edges to: {edges_path}")


def print_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Print a summary of the generated data."""
    num_nodes = len(nodes_df)
    num_edges = len(edges_df)
    stats = calculate_graph_stats(num_nodes, num_edges)
    
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    
    print(f"\n📊 Generated {num_nodes} nodes and {num_edges} edges")
    
    print("\n📋 Department Distribution:")
    dept_counts = nodes_df['department'].value_counts()
    for dept, count in dept_counts.items():
        print(f"   • {dept}: {count} students")
    
    print("\n📅 Batch Distribution:")
    batch_counts = nodes_df['batch'].value_counts().sort_index()
    for batch, count in batch_counts.items():
        print(f"   • {batch}: {count} students")
    
    print("\n🎭 Society Distribution:")
    society_counts = nodes_df['society'].value_counts()
    for society, count in society_counts.items():
        print(f"   • {society}: {count} students")
    
    print("\n" + "="*50)
    print("🔗 GRAPH STATISTICS (Sparsity Verification)")
    print("="*50)
    print(f"   • Average Degree: {stats['average_degree']} (Target: 8-15)")
    print(f"   • Density: {stats['density']} (Target: < 0.05)")
    
    # Verification status
    avg_deg_ok = 8 <= stats['average_degree'] <= 15
    density_ok = stats['density'] < 0.05
    
    print("\n📈 Verification:")
    print(f"   • Average Degree in range [8-15]: {'✅ PASS' if avg_deg_ok else '❌ FAIL'}")
    print(f"   • Density < 0.05: {'✅ PASS' if density_ok else '❌ FAIL'}")
    print("="*50)


if __name__ == "__main__":
    print("🎓 University Social Graph Data Generator (Sparse Graph)")
    print("-" * 50)
    
    # Generate students (nodes)
    print("\n⏳ Generating students...")
    nodes_df = generate_students(NUM_STUDENTS)
    print(f"✓ Generated {len(nodes_df)} students")
    
    # Generate relationships (edges) with strict probabilities
    print("\n⏳ Generating relationships (sparse graph)...")
    print("   Using probabilities:")
    print(f"   • Same society (not 'None'): +{SAME_SOCIETY_PROB*100}%")
    print(f"   • Same batch AND dept: +{SAME_BATCH_DEPT_PROB*100}%")
    print(f"   • Random noise: +{RANDOM_NOISE_PROB*100}%")
    
    edges_df = generate_edges(nodes_df)
    print(f"✓ Generated {len(edges_df)} relationships")
    
    # Save to CSV files
    print("\n⏳ Saving data to CSV files...")
    save_data(nodes_df, edges_df)
    
    # Print summary with graph statistics
    print_summary(nodes_df, edges_df)
