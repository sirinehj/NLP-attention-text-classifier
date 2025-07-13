import tensorflow as tf
from model.attention_lstm import build_model
from utils.preprocess import load_and_preprocess_data
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
import os
from tqdm.keras import TqdmCallback
from keras.callbacks import TensorBoard 

CONFIG = {
    "data_path": "data/training.1600000.processed.noemoticon.csv",
    "vocab_size": 10000,
    "max_len": 50,
    "embedding_dim": 64,
    "batch_size": 256,
    "epochs": 10,
    "output_dir": "model/saved_models",
    "sample_frac": 0.1,
    "patience": 3 
}

callbacks=[
    EarlyStopping(patience=2, restore_best_weights=True),
    TqdmCallback(verbose=1)
],

log_dir = os.path.join(CONFIG["output_dir"], "logs")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def train():
    # 1. Load data
    print("Loading data...")
    X_train, X_val, y_train, y_val, tokenizer = load_and_preprocess_data(
        CONFIG["data_path"],
        num_words=CONFIG["vocab_size"],
        max_len=CONFIG["max_len"],
        sample_frac=CONFIG["sample_frac"]
    )

    # 2. Build model
    print("Building model...")
    model = build_model(  # Fixed function name
        vocab_size=CONFIG["vocab_size"],
        max_len=CONFIG["max_len"],
        embedding_dim=CONFIG["embedding_dim"]
    )
    model.summary()

    # 3. Train
    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1
    )

    # 4. Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Validation Accuracy: {acc:.4f}")

    # 5. Save
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    model.save(f"{CONFIG['output_dir']}/sentiment_attention.h5")
    with open(f"{CONFIG['output_dir']}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    plot_training_history(history)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(f"{CONFIG['output_dir']}/training_history.png")
    plt.close()

if __name__ == "__main__":
    train()