import keras
from keras.layers import Dense, LSTM, Input, Activation, concatenate, Flatten, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Add
from keras.models import Model
import numpy as np

def positional_encoding(max_len, d_model):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  
    return pos_enc

def build_model(shape):
    input_shape = (shape[0], shape[1])
    inputs = Input(shape=input_shape) 
    d_model = input_shape[1]
    pos_enc = positional_encoding(input_shape[0], d_model)
    inputs_pos_enc = inputs + pos_enc
    lstm_out = Bidirectional(LSTM(256, return_sequences=True))(inputs_pos_enc)

    # Multi-Head Self-Attention mechanism
    attention_out, attention_scores = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out, return_attention_scores=True)
    attention_out = LayerNormalization(epsilon=1e-5)(attention_out)

    residual = Add()([lstm_out, attention_out])
    flattened = Flatten()(residual)
    dense1 = Dense(1028, activation='relu')(flattened)
    dropout1 = Dropout(0.1)(dense1)  
    dense2 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(0.1)(dense2)  
    dense3 = Dense(256, activation='relu')(dropout2)
    dropout3 = Dropout(0.1)(dense3)
    output = Dense(1)(dropout3)  

    # Build model
    model = Model(inputs=inputs, outputs=output)
    
    model.summary()
    return model
    
    


