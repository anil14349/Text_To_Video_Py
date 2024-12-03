import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_training_data(
    data_path: str,
    val_split: float = 0.2
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load and split training data."""
    
    # Read CSV file
    df = pd.read_csv(data_path)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
    
    # Extract texts and summaries
    train_texts = train_df['text'].tolist()
    train_summaries = train_df['summary'].tolist()
    val_texts = val_df['text'].tolist()
    val_summaries = val_df['summary'].tolist()
    
    return train_texts, train_summaries, val_texts, val_summaries 