import pandas as pd
from DataLoader import plotTrainTestLoss,x_train,x_val,y_train,y_val
from LSTM import build_model
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf

with tf.device("/GPU:0"):
    DemandModel1 = build_model([2,48])
    # DemandModel1.load_weights('demandModel248Seq/ 30- 0.0016.ckpt')

    checkpoint_lstm = ModelCheckpoint(
        filepath='solarModel2/{epoch: 02d}-{val_loss: .4f}.ckpt',
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    DemandModel1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
    history = DemandModel1.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val),callbacks=[checkpoint_lstm])

    plotTrainTestLoss(history,'solarModel2')





