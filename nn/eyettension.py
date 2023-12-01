import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from transformers import AutoTokenizer, TFDistilBertForSequenceClassification

class eye(keras.Model):

    def __init__(self, dropout = 0.1):
        super().__init__()
        self.distill_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states = True)


        self.dropout_rate = dropout

        # self.input_shape = 100

        # self.transformer_input_id = tf.keras.Input(shape=(input_shape,),dtype='int32')
        # self.transformer_input_att = tf.keras.Input(shape=(input_shape,),dtype='int32')
        # self.transformer_input_mask = tf.keras.Input(shape=(input_shape,),dtype='int32')


        # self.lstm_input = tf.keras.Input(shape=(input_shape, 22),dtype='float32')



        self.lstm1 = Bidirectional(LSTM(256, dropout=self.dropout_rate,  return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(128, dropout=self.dropout_rate))


        self.dense1 = tf.keras.layers.Dense(128,activation='relu')

        self.dense2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense3 = tf.keras.layers.Dense(64,activation='relu')

        self.dense4 = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense5 = tf.keras.layers.Dense(32,activation='relu')

        self.dense6 = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense7 = tf.keras.layers.Dense(1 ,activation='sigmoid')
        # model = tf.keras.models.Model(inputs = [transformer_input_id,transformer_input_att,transformer_input_mask, lstm_input],outputs = output)
        # model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
        # for i in model.layers:
        #     if (i.name.startswith('tf_distil')):
        #         i.trainable = False
        #     else:
        #         i.trainable = True
        # model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics= ['AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    def call(self, inputs):
        output = self.distill_model([inputs[0],inputs[1]])
        output = output.hidden_states[-1]
        # output = np.arange(24).reshape(2,3,4)
        # print(output)

        mask = inputs[2]
        lstm_input = inputs[3]



        merged_word_emb = np.zeros(output.shape)
        # mask = transformer_input_mask
        for d in range(mask.shape[0]):
          for word_idx in range(mask.shape[1]):
            ii = (word_idx == mask[d, :])
            merged_word_emb[d, word_idx, :] = np.mean(output[d, ii, :], axis=0)

        print(merged_word_emb)
        concat = tf.keras.layers.concatenate([merged_word_emb, lstm_input], axis  = 2, name = 'concat')
        out = self.lstm1(concat)
        out = self.lstm2(out)
        out = self.dense1(out)
        out = self.dense2(out)

        out = self.dense3(out)
        out = self.dense4(out)
        out = self.dense5(out)
        out = self.dense6(out)
        out = self.dense7(out)



        return out
