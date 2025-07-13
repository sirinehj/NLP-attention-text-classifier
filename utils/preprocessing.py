import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove @ and #
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def load_and_preprocess_data(file_path, num_words=10000, max_len=50, test_size=0.2):
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    
    # Sentiment140: column 0 is sentiment, column 5 is text
    df = df[[0, 5]]
    df.columns = ['label', 'text']

    # Convert labels: 0 -> negative, 4 -> positive
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    df['text'] = df['text'].apply(clean_text)

    # Tokenize
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    X = padded
    y = df['label'].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    return X_train, X_val, y_train, y_val, tokenizer
