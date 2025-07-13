# 🧠 Sentiment Analysis with LSTM + Attention

This project implements a sentiment analysis model using a Bidirectional LSTM with an attention mechanism. It is trained on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset, which contains 1.6 million tweets labeled as positive or negative.

The goal is to explore how attention mechanisms can enhance text classification by allowing the model to focus on the most relevant parts of an input sequence.

---

## 📂 Project Structure

text-classification-attention/

├── data/ # Contains the Sentiment140 dataset

├── model/

│ ├── attention_lstm.py # Model definition with attention

│ └── saved_models/ # Trained model (.h5), tokenizer, and training plot

├── utils/

│ └── preprocess.py # Text preprocessing and tokenization

├── train.py # Script to train and save the model

├── visualize_attention.py # (optional) Visualization script for attention

├── requirements.txt

└── README.md

-----

## 🛠️ Installation

# Create and activate a virtual environment (recommended)
conda create -n attention-nlp python=3.9
conda activate attention-nlp

# Install dependencies
pip install -r requirements.txt

---

## 🧪 Training
python train.py

The script will:

Preprocess and tokenize the text

Train the model with early stopping

Save the model (.h5) and tokenizer (.pkl)

Plot training history

---

## 🔎 Attention Visualization (Optional)
To inspect what the model focuses on in a sentence, you can run:
python visualize_attention.py
You can modify the sentence variable in the script to try different inputs.

---

## 📊 Results
The model uses:

Embedding layer with masking

Bidirectional LSTM (64 units)

Custom attention mechanism

Dense output layer for binary classification

Training/validation accuracy and loss are plotted in model/saved_models/training_history.png.

---

## 📘 Dataset Info
Source: Sentiment140 - Kaggle

Size: 1.6 million tweets

Labels:

0: Negative sentiment

4: Positive sentiment

Labels were converted to 0 (negative) and 1 (positive) for binary classification.

## ✨ Motivation
This project was developed as a personal learning initiative to better understand attention mechanisms in Natural Language Processing (NLP), particularly how they enhance interpretability and improve performance in sentiment classification tasks involving noisy or informal text like tweets.

---

🧠 Credits
Built with ❤️ by Sirine Hjaij — Based on "Sujet 13: Mécanisme d'attention appliqué à la classification de texte".
