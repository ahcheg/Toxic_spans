import pandas as pd
import numpy as np
import gc
import json
import stanza
from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras import *
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import TFRobertaModel,RobertaTokenizer
import helpers_1
import CRF_keras
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Flatten, Dropout
from keras.models import Model
from keras import Input
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
test_set = pd.read_csv(ROOT_DIR + '/data/tsd_trial.csv')
test_set['spans'] = test_set['spans'].apply(lambda x : json.loads(x))
train_set = pd.read_csv(ROOT_DIR + '/data/tsd_train.csv')
train_set['spans'] = train_set['spans'].apply(lambda x : json.loads(x))
toxic_span_dataset = test_set._append(train_set,ignore_index=True)
toxic_span_dataset['text'] = toxic_span_dataset['text'].apply(lambda x : x.lower())


def createModel(max_input_length,base_model):
   
    input_ids_layer = Input(shape=(max_input_length,),name="encoder_input_ids",dtype=tf.int32)
    input_attention_mask_layer = Input(shape=(max_input_length,),name="encoder_attention_mask",dtype=tf.int32)
    input_length = Input(shape=(1,),name="length",dtype=tf.int32)
    base_model.trainable = True
    base_model = base_model(input_ids_layer,attention_mask=input_attention_mask_layer,return_dict=True)
    output = LSTM(512,return_sequences=True)(base_model.last_hidden_state)
    output = Dense(3,activation="linear")(output)
    crf = CRF_keras.CRFLayer()
    output = crf(inputs=[output,input_length])
    model = Model(inputs=[input_ids_layer,input_attention_mask_layer,input_length],outputs=output)
    model.compile(optimizer=Adam(learning_rate=3e-5),loss=crf.loss,metrics=['accuracy'])
    return model
    
max_length = 400

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

texts = toxic_span_dataset['text'].to_numpy()
targets = helpers_1.createNEROutputs(texts,toxic_span_dataset['spans'],max_length,tokenizer)
all_spans = toxic_span_dataset['spans'].to_numpy()
result_test = []
result_train = []
kf = KFold(n_splits=5)
train_test_indices = []
for train_index,test_index in kf.split(texts):
    train_test_indices.append((train_index,test_index))




train_index,test_index = train_test_indices.pop()
x_train , x_test = list(texts[train_index]) , list(texts[test_index])
y_train , y_test = targets[train_index] , targets[test_index]
model = None
base_model = None
gc.collect()
tf.keras.backend.clear_session()
base_model = TFRobertaModel.from_pretrained('roberta-base')
model = createModel(max_length,base_model)
train_data = helpers_1.createInputForNER(x_train,max_length,tokenizer)
test_data = helpers_1.createInputForNER(x_test,max_length,tokenizer)
spans_test = all_spans[test_index]
spans_train = all_spans[train_index]
model.fit(train_data,y_train,batch_size=16,epochs=2,callbacks=[callbacks.ModelCheckpoint("/content/drive/MyDrive/toxic span/final saved models/NER/roberta/LSTM_crf/ner",save_weights_only=True)])
preds = model.predict(test_data)
indices = helpers_1.createIndicesForNERModel(preds,x_test,tokenizer)
f1_toxic = helpers_1.avg_f1(indices,spans_test)
print("test F1 = %f"%(f1_toxic))
result_test.append(f1_toxic)
preds = model.predict(train_data)
indices = helpers_1.createIndicesForNERModel(preds,x_train,tokenizer)
f1_toxic = helpers_1.avg_f1(indices,spans_train)
print("train F1 = %f"%(f1_toxic))
result_train.append(f1_toxic)
