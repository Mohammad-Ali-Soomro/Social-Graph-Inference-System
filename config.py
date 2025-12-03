"""
Configuration settings for the University Social Graph Inference System
"""

# Data Generation Settings
DATA_CONFIG = {
    "num_students": 500,
    "num_courses": 50,
    "num_clubs": 20,
    "num_dorms": 10,
    "max_courses_per_student": 6,
    "max_clubs_per_student": 3,
    "friendship_probability": 0.1,
}

# Graph Settings
GRAPH_CONFIG = {
    "edge_types": ["same_course", "same_club", "same_dorm", "same_year", "friendship"],
    "edge_weights": {
        "same_course": 0.3,
        "same_club": 0.4,
        "same_dorm": 0.5,
        "same_year": 0.2,
        "friendship": 1.0,
    },
}

# Model Settings
MODEL_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "batch_size": 64,
}

# Training Settings
TRAINING_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "negative_sampling_ratio": 1.0,
    "random_seed": 42,
}

# UI Settings
UI_CONFIG = {
    "page_title": "University Social Graph Inference System",
    "page_icon": "🎓",
    "layout": "wide",
    "graph_height": "600px",
    "graph_width": "100%",
}
