import tensorflow as tf
from keras.layers import Layer, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras import Input

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weights", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, hidden_states):
        score = tf.nn.tanh(tf.matmul(hidden_states, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        return context_vector


def build_model(vocab_size, embedding_dim=128, max_len=50):
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len)(inputs)
    
    # Added dropout for regularization
    x = Bidirectional(LSTM(64, return_sequences=True, 
                       dropout=0.2, recurrent_dropout=0.2))(x)
    
    x = Attention()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']  # Removed precision and recall
    )
    return model
