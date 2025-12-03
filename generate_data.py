"""
Generate synthetic university social graph data.
Creates nodes (students) and edges (relationships) CSVs.
"""

import os
import random
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

# Configuration
NUM_STUDENTS = 500
DEPARTMENTS = ['CS', 'EE', 'BBA', 'Psychology']
BATCHES = [2021, 2022, 2023, 2024]
SOCIETIES = ['Debating', 'ACM', 'Music', 'Sports', 'None']

# Connection probabilities
SAME_SOCIETY_PROB = 0.50      # 50% chance if same society
SAME_BATCH_DEPT_PROB = 0.30   # 30% chance if same batch AND department
RANDOM_NOISE_PROB = 0.05      # 5% random noise connections


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
    Generate edges (relationships) between students based on rules.
    
    Rules:
    - 50% chance if same society (excluding 'None')
    - 30% chance if same batch AND department
    - 5% random noise connections
    
    Args:
        students_df: DataFrame containing student information
        
    Returns:
        DataFrame with edge connections (source, target)
    """
    edges = set()  # Use set to avoid duplicate edges
    num_students = len(students_df)
    
    # Convert DataFrame to list of dicts for faster access
    students = students_df.to_dict('records')
    
    # Iterate through all pairs of students
    for i in range(num_students):
        for j in range(i + 1, num_students):
            student_a = students[i]
            student_b = students[j]
            
            connected = False
            
            # Rule 1: Same society (50% chance, excluding 'None')
            if (student_a['society'] == student_b['society'] and 
                student_a['society'] != 'None'):
                if random.random() < SAME_SOCIETY_PROB:
                    connected = True
            
            # Rule 2: Same batch AND department (30% chance)
            if (student_a['batch'] == student_b['batch'] and 
                student_a['department'] == student_b['department']):
                if random.random() < SAME_BATCH_DEPT_PROB:
                    connected = True
            
            # Rule 3: Random noise (5% chance)
            if not connected and random.random() < RANDOM_NOISE_PROB:
                connected = True
            
            if connected:
                # Store edge as tuple (smaller_id, larger_id) to avoid duplicates
                edge = (student_a['student_id'], student_b['student_id'])
                edges.add(edge)
    
    # Convert to DataFrame
    edges_df = pd.DataFrame(list(edges), columns=['source', 'target'])
    
    return edges_df


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
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    
    print(f"\n📊 Generated {len(nodes_df)} nodes and {len(edges_df)} edges")
    
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
    
    print("\n🔗 Edge Statistics:")
    avg_connections = (2 * len(edges_df)) / len(nodes_df)
    print(f"   • Average connections per student: {avg_connections:.2f}")
    print("="*50)


if __name__ == "__main__":
    print("🎓 University Social Graph Data Generator")
    print("-" * 40)
    
    # Generate students (nodes)
    print("\n⏳ Generating students...")
    nodes_df = generate_students(NUM_STUDENTS)
    print(f"✓ Generated {len(nodes_df)} students")
    
    # Generate relationships (edges)
    print("\n⏳ Generating relationships...")
    edges_df = generate_edges(nodes_df)
    print(f"✓ Generated {len(edges_df)} relationships")
    
    # Save to CSV files
    print("\n⏳ Saving data to CSV files...")
    save_data(nodes_df, edges_df)
    
    # Print summary
    print_summary(nodes_df, edges_df)
