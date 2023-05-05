import pandas as pd
import numpy as np
import gc
import json
import stanza
from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras import *
import tensorflow.keras.backend as K
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


def createNEROutputs(texts,spans,max_length,tokenizer):
    outputs = []
    for text,span in zip(texts,spans):
        output = np.zeros(max_length*3,dtype=np.float).reshape((max_length,3))
        tokens = tokenizer.tokenize(text)[:max_length]
        length = 0
        start = True
        for i in range(len(tokens),max_length):
            output[i,0] = 1.0
        for index,token in enumerate(tokens):
            sub = True
            if "Ġ" in token:
                sub = False
                token = token[1:]
            if not start:
                next_index = text[length:].find(token)
                if next_index == 0:
                    sub = True
                length += next_index
            # if length in span and not sub:
            #     output[index,2] = 1.0
            #     output[index,0] = 0.0
            if length in span:
                output[index,2] = 1.0
                output[index,0] = 0.0
            else:
                output[index,1] = 1.0
                output[index,0] = 0.0
            length += len(token)
            start = False
        outputs.append(output)
    return np.array(outputs)
     

def NERGetIndicesSingleText(outputs,text,tokenizer):
    outputs = tf.argmax(outputs,axis=-1)
    tokens = tokenizer.tokenize(text)
    index = 0
    indexes = []
    sub = False
    prev = False
    for token,output in zip(tokens,outputs):
        if token[0] == "Ġ":
            token = token[1:]
            sub = False
        elif token.isalpha():
            sub = True
        else:
            sub = False
        temp_index = text[index:].find(token)
        temp_start = index+temp_index
        if output == 2 or (sub and prev and output != 0):
            prev = True
            indexes = indexes + list(range(temp_start,temp_start+len(token)))
        else:
            prev = False
        index = temp_start+len(token)
    return np.array(indexes)
     

def createIndicesForNERModel(predicts,texts,tokenizer):
    outputs = []
    for text,pred in zip(texts,predicts):
         indices = NERGetIndicesSingleText(pred,text,tokenizer)
         outputs.append(indices)
    return outputs
     

def f1(preds,trues):
    if len(trues) == 0:
        return 1. if len(preds) == 0 else 0.
    if len(preds) == 0:
        return 0.
    predictions_set = set(preds)
    gold_set = set(trues)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)
     

def avg_f1(preds,trues):
    avg_f1_total = 0.0
    for pred,true in zip(preds,trues):
        avg_f1_total += f1(pred,true)
    return avg_f1_total/len(preds)
     
     

def createInputForNER(texts,max_length,tokenizer):
    input_length = []
    for text in texts:
        input_length.append(min(max_length,len(tokenizer.tokenize(text))))
    tokens = tokenizer(texts,padding="max_length",max_length=max_length,return_tensors="tf",truncation=True)
    data = [np.array(tokens['input_ids']),np.array(tokens['attention_mask']),np.array(input_length)]
    return data