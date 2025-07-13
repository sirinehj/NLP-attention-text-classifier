import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\w\s.!?,;]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(file_path, num_words=10000, max_len=50, test_size=0.2, sample_frac=1.0):
    # Read data
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    
    # Take a subset of data for faster experimentation
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Sentiment140: column 0 is sentiment, column 5 is text
    df = df[[0, 5]]
    df.columns = ['label', 'text']

    # Convert labels: 0 -> negative, 4 -> positive
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Check class balance
    print("\nClass distribution:")
    print(df['label'].value_counts())

    # Tokenization
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Train-test split
    X = padded
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=42
    )

    return X_train, X_val, y_train, y_val, tokenizer