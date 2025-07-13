import tensorflow as tf
from model.attention_lstm import build_model
from utils.preprocess import load_and_preprocess_data
from keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                           TensorBoard, ReduceLROnPlateau)
import matplotlib.pyplot as plt
import pickle
import os
import time
from datetime import datetime
from tqdm.keras import TqdmCallback
import numpy as np

# Configuration with additional parameters
CONFIG = {
    "data_path": "data/training.1600000.processed.noemoticon.csv",
    "vocab_size": 10000,
    "max_len": 50,
    "embedding_dim": 64,
    "batch_size": 256,
    "epochs": 15,  # Increased with early stopping
    "output_dir": "model/saved_models",
    "sample_frac": 0.1,
    "patience": 3,  # More tolerant early stopping
    "min_delta": 0.001,  # Minimum improvement required
    "reduce_lr_patience": 2,  # Patience for LR reduction
    "reduce_lr_factor": 0.5,  # LR reduction factor
    "min_lr": 1e-6  # Minimum learning rate
}

def setup_callbacks():
    """Configure and return training callbacks"""
    log_dir = os.path.join(CONFIG["output_dir"], "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG["patience"],
            min_delta=CONFIG["min_delta"],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG["output_dir"], 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CONFIG["reduce_lr_factor"],
            patience=CONFIG["reduce_lr_patience"],
            min_lr=CONFIG["min_lr"],
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch='500,520'
        ),
        TqdmCallback(verbose=1)
    ]

def train():
    try:
        # 1. Setup and data loading
        start_time = time.time()
        print("🚀 Initializing training...")
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        
        print("📊 Loading and preprocessing data...")
        X_train, X_val, y_train, y_val, tokenizer = load_and_preprocess_data(
            CONFIG["data_path"],
            num_words=CONFIG["vocab_size"],
            max_len=CONFIG["max_len"],
            sample_frac=CONFIG["sample_frac"]
        )
        print(f"✅ Data loaded. Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # 2. Model building
        print("\n🛠️ Building model...")
        model = build_model(
            vocab_size=CONFIG["vocab_size"],
            max_len=CONFIG["max_len"],
            embedding_dim=CONFIG["embedding_dim"]
        )
        model.summary()
        
        # 3. Training
        print("\n🎯 Starting training...")
        callbacks = setup_callbacks()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            callbacks=callbacks,
            verbose=0  # Using Tqdm instead
        )
        
        # 4. Evaluation
        print("\n📈 Evaluating model...")
        results = model.evaluate(X_val, y_val, verbose=0)
        
        print("\n🏆 Final Metrics:")
        print(f"Validation Loss: {results[0]:.4f}")
        print(f"Validation Accuracy: {results[1]:.4f}")
        if len(results) > 2:  # If additional metrics exist
            print(f"Validation Precision: {results[2]:.4f}")
            print(f"Validation Recall: {results[3]:.4f}")
        
        # 5. Save artifacts
        print("\n💾 Saving model and artifacts...")
        model.save(os.path.join(CONFIG["output_dir"], 'final_model.h5'))
        with open(os.path.join(CONFIG["output_dir"], 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Save training history
        plot_training_history(history)
        save_training_report(history, results)
        
        print(f"\n⏱️ Total training time: {(time.time() - start_time)/60:.2f} minutes")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

def plot_training_history(history):
    """Visualize and save training metrics"""
    plt.figure(figsize=(15, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy', pad=20)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss', pad=20)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'training_history.png'), dpi=300)
    plt.close()

def save_training_report(history, results):
    """Save detailed training report"""
    report = {
        'config': CONFIG,
        'final_metrics': {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]) if len(results) > 2 else None,
            'recall': float(results[3]) if len(results) > 2 else None
        },
        'training_history': history.history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(CONFIG["output_dir"], 'training_report.pkl'), 'wb') as f:
        pickle.dump(report, f)

if __name__ == "__main__":
    train()