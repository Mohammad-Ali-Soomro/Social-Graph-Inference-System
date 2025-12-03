"""
Streamlit Dashboard for University Social Graph Inference System.
Visualizes the social network and provides AI-powered friend recommendations.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network
import tempfile

from graph_utils import load_graph, get_graph_stats, get_centrality_measures, get_node_info
from inference import (
    load_trained_model, 
    load_encoders, 
    recommend_friends,
    get_connection_probability
)
from train_model import prepare_node_features, networkx_to_pyg


# Page configuration
st.set_page_config(
    page_title="University Social Graph Inference",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color mapping for societies
SOCIETY_COLORS = {
    'Debating': '#FF6B6B',    # Red
    'ACM': '#4ECDC4',          # Teal
    'Music': '#45B7D1',        # Blue
    'Sports': '#96CEB4',       # Green
    'None': '#CCCCCC'          # Gray
}

DEPARTMENT_COLORS = {
    'CS': '#FF6B6B',
    'EE': '#4ECDC4',
    'BBA': '#45B7D1',
    'Psychology': '#96CEB4'
}


@st.cache_resource
def load_all_resources():
    """Load and cache all resources (graph, model, encoders)."""
    # Load graph
    graph = load_graph()
    
    # Convert all node IDs to Python int to avoid numpy.int64 issues
    # Create a new graph with proper int node IDs
    import networkx as nx
    G_clean = nx.Graph()
    
    for node in graph.nodes():
        node_int = int(node)
        attrs = graph.nodes[node]
        G_clean.add_node(node_int, **attrs)
    
    for edge in graph.edges():
        G_clean.add_edge(int(edge[0]), int(edge[1]))
    
    # Load model and encoders
    model, checkpoint = load_trained_model()
    encoders = load_encoders()
    
    # Prepare features and data
    node_features, _ = prepare_node_features(G_clean, fit_encoders=False, encoders=encoders)
    data = networkx_to_pyg(G_clean, node_features)
    node_mapping = data.node_mapping
    
    return G_clean, model, data, node_mapping, checkpoint


def create_pyvis_network(graph, highlight_student=None, show_only_neighbors=False, color_by='society'):
    """
    Create a Pyvis network visualization.
    
    Args:
        graph: NetworkX graph
        highlight_student: Student ID to highlight
        show_only_neighbors: If True, only show the highlighted student and neighbors
        color_by: 'society' or 'department'
        
    Returns:
        Pyvis Network object
    """
    # Create Pyvis network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Physics settings for better visualization
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 150
            }
        },
        "nodes": {
            "font": {"size": 12}
        },
        "edges": {
            "color": {"inherit": true},
            "smooth": false
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    color_map = SOCIETY_COLORS if color_by == 'society' else DEPARTMENT_COLORS
    
    # Determine which nodes to show
    if show_only_neighbors and highlight_student:
        neighbors = [int(n) for n in graph.neighbors(highlight_student)]
        nodes_to_show = set([int(highlight_student)]) | set(neighbors)
    else:
        # Sample nodes for performance (show max 150 nodes)
        all_nodes = [int(n) for n in graph.nodes()]
        if len(all_nodes) > 150:
            import random
            random.seed(42)
            nodes_to_show = set(random.sample(all_nodes, 150))
            if highlight_student:
                nodes_to_show.add(int(highlight_student))
                neighbors = [int(n) for n in graph.neighbors(highlight_student)][:20]
                nodes_to_show.update(neighbors)
        else:
            nodes_to_show = set(all_nodes)
    
    # Add nodes
    for node in nodes_to_show:
        attrs = graph.nodes[node]
        
        # Get color based on attribute
        if color_by == 'society':
            color = color_map.get(attrs.get('society', 'None'), '#CCCCCC')
        else:
            color = color_map.get(attrs.get('dept', 'Unknown'), '#CCCCCC')
        
        # Highlight selected student
        if node == highlight_student:
            size = 30
            border_width = 3
            border_color = '#FFD700'  # Gold border
        else:
            size = 15
            border_width = 1
            border_color = color
        
        # Create tooltip
        title = f"""
        <b>{attrs.get('name', 'Unknown')}</b><br>
        ID: {node}<br>
        Department: {attrs.get('dept', 'Unknown')}<br>
        Batch: {attrs.get('batch', 'Unknown')}<br>
        Society: {attrs.get('society', 'Unknown')}
        """
        
        # Ensure node ID is int for Pyvis
        node_id = int(node)
        
        net.add_node(
            node_id,
            label=str(node_id),
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            borderWidthSelected=3
        )
    
    # Add edges (only between visible nodes)
    for edge in graph.edges():
        edge_0 = int(edge[0])
        edge_1 = int(edge[1])
        if edge_0 in nodes_to_show and edge_1 in nodes_to_show:
            # Highlight edges connected to selected student
            if highlight_student and (edge_0 == int(highlight_student) or edge_1 == int(highlight_student)):
                net.add_edge(edge_0, edge_1, color='#FFD700', width=2)
            else:
                net.add_edge(edge_0, edge_1, color='#CCCCCC', width=0.5)
    
    return net


def render_pyvis_network(net):
    """Render Pyvis network in Streamlit."""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        temp_path = f.name
    
    # Read HTML content
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Clean up temp file
    os.unlink(temp_path)
    
    # Render in Streamlit
    components.html(html_content, height=620, scrolling=True)


def main():
    """Main application function."""
    
    # Title
    st.title("🎓 University Social Graph Inference")
    st.markdown("*Predict hidden relationships between students using Graph Neural Networks*")
    
    # Load resources
    with st.spinner("Loading resources..."):
        graph, model, data, node_mapping, checkpoint = load_all_resources()
    
    # ==================== SIDEBAR ====================
    st.sidebar.header("📊 Graph Statistics")
    
    # Display graph stats
    stats = get_graph_stats(graph)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total Nodes", stats['total_nodes'])
    col2.metric("Total Edges", f"{stats['total_edges']:,}")
    
    col3, col4 = st.sidebar.columns(2)
    col3.metric("Avg Degree", stats['average_degree'])
    col4.metric("Density", stats['density'])
    
    st.sidebar.markdown("---")
    
    # Model info
    st.sidebar.header("🧠 Model Info")
    st.sidebar.metric("Test AUC", f"{checkpoint['test_auc']:.4f}")
    st.sidebar.metric("Test Accuracy", f"{checkpoint['test_acc']:.2%}")
    
    st.sidebar.markdown("---")
    
    # Student selection
    st.sidebar.header("👤 Select Student")
    
    # Get all student IDs and names for the dropdown
    student_options = []
    for node in sorted(graph.nodes()):
        name = graph.nodes[node].get('name', 'Unknown')
        student_options.append(f"{node} - {name}")
    
    selected_option = st.sidebar.selectbox(
        "Choose a student:",
        student_options,
        index=0
    )
    
    # Extract student ID from selection
    selected_student_id = int(selected_option.split(" - ")[0])
    
    # Show selected student info
    student_info = get_node_info(graph, selected_student_id)
    
    st.sidebar.markdown("**Selected Student Details:**")
    st.sidebar.markdown(f"- **Name:** {student_info['name']}")
    st.sidebar.markdown(f"- **Department:** {student_info['department']}")
    st.sidebar.markdown(f"- **Batch:** {student_info['batch']}")
    st.sidebar.markdown(f"- **Society:** {student_info['society']}")
    st.sidebar.markdown(f"- **Current Friends:** {student_info['num_connections']}")
    
    # ==================== MAIN AREA ====================
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌐 Network Visualization", "🤖 AI Recommendations", "📈 Analytics"])
    
    # ==================== TAB 1: Network Visualization ====================
    with tab1:
        st.header("Network Visualization")
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color_by = st.selectbox(
                "Color nodes by:",
                ["society", "department"],
                format_func=lambda x: x.title()
            )
        
        with col2:
            show_neighbors_only = st.checkbox(
                "Show only selected student's network",
                value=True
            )
        
        with col3:
            st.markdown("**Legend:**")
            if color_by == 'society':
                for society, color in SOCIETY_COLORS.items():
                    st.markdown(f"<span style='color:{color}'>●</span> {society}", unsafe_allow_html=True)
            else:
                for dept, color in DEPARTMENT_COLORS.items():
                    st.markdown(f"<span style='color:{color}'>●</span> {dept}", unsafe_allow_html=True)
        
        # Create and render network
        with st.spinner("Generating network visualization..."):
            net = create_pyvis_network(
                graph, 
                highlight_student=selected_student_id,
                show_only_neighbors=show_neighbors_only,
                color_by=color_by
            )
            render_pyvis_network(net)
        
        st.info("💡 **Tip:** Hover over nodes to see student details. The selected student is highlighted with a gold border.")
    
    # ==================== TAB 2: AI Recommendations ====================
    with tab2:
        st.header("🤖 AI Friend Recommendations")
        st.markdown(f"**Recommendations for:** {student_info['name']} (ID: {selected_student_id})")
        
        # Number of recommendations slider
        top_k = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)
        
        # Get recommendations
        with st.spinner("Generating AI recommendations..."):
            recommendations = recommend_friends(
                student_id=selected_student_id,
                model=model,
                data=data,
                graph=graph,
                node_mapping=node_mapping,
                top_k=top_k
            )
        
        if recommendations:
            # Display recommendations in a nice format
            st.success(f"Found {len(recommendations)} potential friends!")
            
            # Create columns for cards
            for i, rec in enumerate(recommendations):
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"### {i+1}. {rec['name']}")
                        st.markdown(f"**ID:** {rec['student_id']}")
                    
                    with col2:
                        st.markdown(f"**Department:** {rec['department']}")
                        st.markdown(f"**Batch:** {rec['batch']}")
                        st.markdown(f"**Society:** {rec['society']}")
                    
                    with col3:
                        # Similarity score with progress bar
                        st.metric("Similarity", rec['confidence'])
                        st.progress(rec['similarity_score'])
                    
                    st.markdown("---")
            
            # Also show as a table
            st.subheader("📋 Recommendations Table")
            rec_df = pd.DataFrame(recommendations)
            rec_df.index = rec_df.index + 1
            rec_df = rec_df.rename(columns={
                'student_id': 'Student ID',
                'name': 'Name',
                'department': 'Department',
                'batch': 'Batch',
                'society': 'Society',
                'similarity_score': 'Score',
                'confidence': 'Confidence'
            })
            st.dataframe(rec_df, use_container_width=True)
            
        else:
            st.warning("No recommendations found. This student may already be connected to most similar students.")
    
    # ==================== TAB 3: Analytics ====================
    with tab3:
        st.header("📈 Graph Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Most Connected Students")
            centrality_df = get_centrality_measures(graph, top_n=10)
            centrality_df = centrality_df.rename(columns={
                'student_id': 'ID',
                'name': 'Name',
                'department': 'Dept',
                'batch': 'Batch',
                'society': 'Society',
                'degree_centrality': 'Centrality',
                'connections': 'Connections'
            })
            st.dataframe(centrality_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Distribution Charts")
            
            # Society distribution
            nodes_df = pd.read_csv('data/nodes.csv')
            
            society_counts = nodes_df['society'].value_counts()
            st.bar_chart(society_counts)
            st.caption("Students per Society")
            
            # Department distribution
            dept_counts = nodes_df['department'].value_counts()
            st.bar_chart(dept_counts)
            st.caption("Students per Department")
        
        # Connection probability checker
        st.subheader("🔗 Connection Probability Checker")
        st.markdown("Check the predicted probability of connection between any two students.")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            student1 = st.selectbox("Student 1:", student_options, key="prob_student1")
            student1_id = int(student1.split(" - ")[0])
        
        with col2:
            student2 = st.selectbox("Student 2:", student_options, index=1, key="prob_student2")
            student2_id = int(student2.split(" - ")[0])
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            check_button = st.button("Check Probability", type="primary")
        
        if check_button:
            if student1_id == student2_id:
                st.warning("Please select two different students.")
            else:
                prob = get_connection_probability(
                    student1_id, student2_id, model, data, node_mapping
                )
                
                # Check if already connected
                is_connected = graph.has_edge(student1_id, student2_id)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Connection Probability", f"{prob:.2%}")
                    st.progress(prob)
                with col2:
                    if is_connected:
                        st.success("✅ These students are already connected!")
                    else:
                        if prob > 0.7:
                            st.info("🎯 High probability! These students should connect.")
                        elif prob > 0.5:
                            st.info("📊 Moderate probability of connection.")
                        else:
                            st.info("📉 Low probability of connection.")


if __name__ == "__main__":
    main()
