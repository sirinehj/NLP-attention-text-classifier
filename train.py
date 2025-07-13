from utils.preprocessing import load_and_preprocess_data
from model.attention_lstm import build_model
from keras.callbacks import EarlyStopping
import os

DATA_PATH = os.path.join("data", "training.1600000.processed.noemoticon.csv")
VOCAB_SIZE = 10000
MAX_LEN = 50
BATCH_SIZE = 64
EPOCHS = 5

X_train, X_val, y_train, y_val, tokenizer = load_and_preprocess_data(DATA_PATH, num_words=VOCAB_SIZE, max_len=MAX_LEN)

model = build_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"✅ Validation Accuracy: {acc:.4f}")

model.save("sentiment_attention_model.h5")
print("✅ Model saved as sentiment_attention_model.h5")
