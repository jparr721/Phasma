import os

from dataloader import load_datasets

if __name__ == "__main__":
    datasets_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "datasets"
    )
    loaded_datasets = load_datasets(datasets_path)
