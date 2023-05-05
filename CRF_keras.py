import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, InputSpec
import tensorflow_addons as tfa
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
try:
    from tensorflow.contrib.crf import crf_decode
except ImportError:
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import array_ops, gen_array_ops, math_ops, rnn, rnn_cell


    class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
        def __init__(self, transition_params):
            self._transition_params = array_ops.expand_dims(transition_params, 0)
            self._num_tags = transition_params.get_shape()[0]

        @property
        def state_size(self):
            return self._num_tags

        @property
        def output_size(self):
            return self._num_tags

        def __call__(self, inputs, state, scope=None):
            state = array_ops.expand_dims(state, 2)  # [B, O, 1]
            transition_scores = state + self._transition_params  # [B, O, O]
            new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
            backpointers = math_ops.argmax(transition_scores, 1)
            backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)  # [B, O]
            return backpointers, new_state


    class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
        def __init__(self, num_tags):
            self._num_tags = num_tags

        @property
        def state_size(self):
            return 1

        @property
        def output_size(self):
            return 1

        def __call__(self, inputs, state, scope=None):
            state = array_ops.squeeze(state, axis=[1])  # [B]
            batch_size = array_ops.shape(inputs)[0]
            b_indices = math_ops.range(batch_size)  # [B]
            indices = array_ops.stack([b_indices, state], axis=1)  # [B, 2]
            new_tags = array_ops.expand_dims(
                gen_array_ops.gather_nd(inputs, indices),  # [B]
                axis=-1)  # [B, 1]

            return new_tags, new_tags


    def crf_decode(potentials, transition_params, sequence_length):
        num_tags = potentials.get_shape()[2]

        # Computes forward decoding. Get last score and backpointers.
        crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
        initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
        inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
        backpointers, last_score = rnn.dynamic_rnn(
            crf_fwd_cell,
            inputs=inputs,
            sequence_length=sequence_length - 1,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)  # [B, T - 1, O], [B, O]
        backpointers = gen_array_ops.reverse_sequence(backpointers, sequence_length - 1, seq_dim=1)  # [B, T-1, O]

        # Computes backward decoding. Extract tag indices from backpointers.
        crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
        initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1), dtype=dtypes.int32)  # [B]
        initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
        decode_tags, _ = rnn.dynamic_rnn(
            crf_bwd_cell,
            inputs=backpointers,
            sequence_length=sequence_length - 1,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)  # [B, T - 1, 1]
        decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
        decode_tags = array_ops.concat([initial_state, decode_tags], axis=1)  # [B, T]
        decode_tags = gen_array_ops.reverse_sequence(decode_tags, sequence_length, seq_dim=1)  # [B, T]

        best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
        return decode_tags, best_score


class CRFLayer(Layer):

    def __init__(self, transition_params=None, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.transition_params = transition_params
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape[0]) == 3

        return input_shape[0]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        n_steps = input_shape[0][1]
        n_classes = input_shape[0][2]
        assert n_steps is None or n_steps >= 2

        self.transition_params = self.add_weight(shape=(n_classes, n_classes),
                                                 initializer='uniform',
                                                 name='transition')
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, n_classes)),
                           InputSpec(dtype='int32', shape=(None, 1))]
        self.built = True

    def viterbi_decode(self, potentials, sequence_length):
        decode_tags, best_score = crf_decode(potentials, self.transition_params, sequence_length)
        return decode_tags

    def call(self, inputs, mask=None, **kwargs):
        inputs, sequence_lengths = inputs
        self.sequence_lengths = K.flatten(sequence_lengths)
        y_pred = self.viterbi_decode(inputs, self.sequence_lengths)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)

        return K.in_train_phase(inputs, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
        log_likelihood, self.transition_params = tfa.text.crf.crf_log_likelihood(
            y_pred, y_true, self.sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_config(self):
        config = {
            'transition_params': K.eval(self.transition_params),
        }
        base_config = super(CRFLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    instanceHolder = {'instance': None}

    class ClassWrapper(CRFLayer):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    return {'CRFLayer': ClassWrapper, 'loss': loss}