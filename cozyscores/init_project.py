# init_project.py - Script to initialize a new project structure for CozyScores
import os
import argparse
from pathlib import Path

def init_project(name):
    root = Path("projects") / name
    folders = ["data/scored_photos", "data/unscored_photos", "results"]
    
    for f in folders:
        os.makedirs(root / f, exist_ok=True)
    
    config_content = f'project_name: "{name}"\n'
    with open(root / "config.yaml", "w") as f:
        f.write(config_content)
        
    print(f"Project {name} initialized at {root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    init_project(parser.parse_args().name)