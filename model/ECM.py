# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 19:08:08
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-25 18:14:52

from tensorflow.contrib.rnn import RNNCell

import tensorflow as tf
import collections


ECMState = collections.namedtuple(
    "ECMState", ("cell_states", "h", "context", "internal_memory"))


class ECMWrapper(RNNCell):
    """
    Emotion Chatting Machine: H. Zhou, et al. AAAI 2018
    (https://arxiv.org/abs/1704.01074)
    Emotion Category Embedding, Internal and External Memory Modules
        cell: vanilla multi-layer RNNCell
        memory: [batch_size, max_time, num_units]
        emo_cat_embs: category embeddings, [batch_size, emo_cat_units]
        emo_cat: emotion category, [batch_size]
        emo_int_units: dimension of internal emotion memory
    """
    def __init__(self, cell, memory, dec_init_states, num_hidden,
                 num_units, dtype, emo_cat_embs, emo_cat, num_emo,
                 emo_int_units, emo_init=None):
        self._cell = cell
        self._memory = memory
        self.num_hidden = num_hidden

        self._dec_init_states = dec_init_states
        self._state_size = ECMState(self._cell.state_size,
                                    num_units, memory.shape[-1].value,
                                    emo_int_units)
        self._num_units = num_units
        self._dtype = dtype

        # ECM hyperparameters
        self._emo_cat_embs = emo_cat_embs
        self._emo_cat = emo_cat
        self._emo_int_units = emo_int_units

        # internal memory
        if emo_init is None:
            initializer = tf.contrib.layers.xavier_initializer()

        self.int_memory = tf.Variable(
            initializer(shape=(num_emo, emo_int_units)),
            name="emo_memory", dtype=dtype)

        self.read_g = tf.layers.Dense(
            emo_int_units, use_bias=False, name="internal_read_gate")
        self.write_g = tf.layers.Dense(
            emo_int_units, use_bias=False, name="internal_write_gate")

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units

    def initial_state(self):
        """
        Generate initial state for ECM wrapped rnn cell
            dec_init_states: None (no states pass), or encoder final states
            num_units: decoder's num of cell units
        Returns:
            h_0: [batch_size, num_units]
            context_0: [batch_size, num_units]
            M_emo_0: [batch_size, emo_int_units]
        """
        h_0 = tf.zeros([1, self._num_units], self._dtype)
        context_0 = self._compute_context(h_0)
        h_0 = context_0 * 0
        M_emo_0 = tf.gather(self.int_memory, self._emo_cat)

        if self._dec_init_states is None:
            batch_size = tf.shape(self._memory)[0]
            cell_states = self._cell.zero_state(batch_size, self._dtype)
        else:
            cell_states = self._dec_init_states

        ecm_state_0 = ECMState(cell_states, h_0, context_0, M_emo_0)

        return ecm_state_0

    def _compute_context(self, query):
        """
        Compute attn scores and weighted sum of memory as the context
            query: [batch_size, num_units]
        Returns:
            context: [batch_size, num_units]
        """
        query = tf.expand_dims(query, -2)
        Wq = tf.layers.dense(query, self.num_hidden, use_bias=False)
        Wm = tf.layers.dense(self._memory, self.num_hidden, use_bias=False)
        e = tf.layers.dense(tf.nn.tanh(Wm + Wq), 1, use_bias=False)
        attn_scores = tf.expand_dims(tf.nn.softmax(tf.squeeze(e, axis=-1)), -1)

        context = tf.reduce_sum(attn_scores * self._memory, axis=1)

        return context

    def _read_internal_memory(self, M_emo, read_inputs):
        """
        Read the internal memory
            M_emo: [batch_size, emo_int_units]
            read_inputs: [batch_size, d]
        Returns:
            M_read: [batch_size, emo_int_units]
        """
        gate_read = tf.nn.sigmoid(self.read_g(read_inputs))
        return (M_emo * gate_read)

    def _write_internal_memory(self, M_emo, new_h):
        """
        Write the internal memory
            M_emo: [batch_size, emo_int_units]
            new_h: [batch_size, num_units]
        Returns:
            M_write: [batch_size, emo_int_units]
        """
        gate_write = tf.nn.sigmoid(self.write_g(new_h))
        return (M_emo * gate_write)

    def __call__(self, inputs, ecm_states):
        """
            inputs: emebeddings of previous word
            states: (cell_states, outputs, context, int_memory)
        """
        prev_cell_states, h, context, M_emo = ecm_states

        # read internal memory
        read_inputs = tf.concat([inputs, h, context], axis=-1)
        M_read = self._read_internal_memory(M_emo, read_inputs)

        # pass into RNN_cell to get the output
        x = [inputs, h, context, self._emo_cat_embs, M_read]
        x = tf.concat(x, axis=-1)
        new_h, cell_states = self._cell.__call__(x, prev_cell_states)

        # update states
        new_M_emo = self._write_internal_memory(M_emo, new_h)
        new_context = self._compute_context(new_h)
        new_ecm_states = ECMState(cell_states, new_h, new_context, new_M_emo)

        return (new_h, new_ecm_states)
