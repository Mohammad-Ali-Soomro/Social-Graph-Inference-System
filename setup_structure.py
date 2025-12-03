"""
Setup script to create the project folder structure.
Creates 'data' and 'models' directories for organizing project files.
"""

import os


def create_project_structure():
    """Create the necessary project directories."""
    
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define folders to create
    folders = [
        "data",    # For storing generated/processed data
        "models"   # For storing trained model files
    ]
    
    print("Setting up project structure...")
    print(f"Base directory: {base_dir}\n")
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"✓ Created folder: {folder}/")
        else:
            print(f"• Folder already exists: {folder}/")
    
    print("\n✓ Project structure setup complete!")


if __name__ == "__main__":
    create_project_structure()
