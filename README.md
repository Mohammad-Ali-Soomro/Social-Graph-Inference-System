# University Social Graph Inference System

A system to predict hidden relationships between students using Graph Neural Networks.

## Tech Stack

- **Data Generation**: Faker (synthetic data)
- **Graph Logic**: NetworkX
- **ML/AI**: PyTorch + PyTorch Geometric (GraphSAGE)
- **UI**: Streamlit + Pyvis

## Project Structure

```
Social Graph Inference System/
├── data/
│   └── data_generator.py      # Synthetic student data generation
├── graph/
│   └── graph_builder.py       # NetworkX graph construction
├── models/
│   └── graphsage.py           # GraphSAGE model implementation
├── training/
│   └── trainer.py             # Model training logic
├── ui/
│   └── app.py                 # Streamlit application
├── utils/
│   └── helpers.py             # Utility functions
├── requirements.txt
├── config.py                  # Configuration settings
└── main.py                    # Entry point
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run ui/app.py
   ```

## Features

- Generate synthetic university student data
- Build social graphs based on shared attributes (courses, clubs, dorms, etc.)
- Train GraphSAGE model to learn node embeddings
- Predict hidden/potential relationships between students
- Interactive visualization with Pyvis
