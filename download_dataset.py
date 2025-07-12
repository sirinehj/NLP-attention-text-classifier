import os
import kaggle
import pandas as pd
from zipfile import ZipFile

# 1. Download dataset using Kaggle API
def download_dataset():
    try:
        # Using the official Kaggle API
        os.system('kaggle datasets download -d kazanova/sentiment140 -p data --unzip')
        print("Download completed successfully!")
    except Exception as e:
        print(f"Download failed. Please manually download from:\n"
              f"https://www.kaggle.com/datasets/kazanova/sentiment140\n"
              f"and place ZIP file in 'data/' folder")
        return False
    return True

# 2. Process the dataset
def process_data():
    try:
        # Load the CSV with proper encoding
        df = pd.read_csv(
            "data/training.1600000.processed.noemoticon.csv",
            encoding='latin1',
            header=None,
            names=['label', 'id', 'date', 'flag', 'user', 'text']
        )
        
        # Convert labels (0=negative, 4=positive) to (0,1)
        df['label'] = df['label'].replace(4, 1)
        
        # Save cleaned version
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/sentiment140_cleaned.csv", index=False)
        
        print(f"Processed {len(df)} tweets")
        print("Saved to: data/processed/sentiment140_cleaned.csv")
        return True
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    if download_dataset():
        process_data()