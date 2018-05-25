# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 19:07:25
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-24 17:11:25


from tensorflow.contrib.rnn import RNNCell

import tensorflow as tf
import collections


AttnState = collections.namedtuple(
    "AttnState", ("cell_states", "h", "context"))


class AttentionWrapper(RNNCell):
    """
    Attention Wrapper: wrap a rnn_cell with attention mechanism (Bahda 2015)
        cell: vanilla multi-layer RNNCell
        memory: [batch_size, max_time, num_units]
    """
    def __init__(self, cell, memory, dec_init_states, num_hidden,
                 num_units, dtype):
        self._cell = cell
        self._memory = memory
        self.num_hidden = num_hidden

        self._dec_init_states = dec_init_states
        self._state_size = AttnState(self._cell.state_size,
                                     num_units, memory.shape[-1].value)
        self._num_units = num_units
        self._dtype = dtype

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units

    def initial_state(self):
        """
        Generate initial state for attn wrapped rnn cell
            dec_init_states: None (no states pass), or encoder final states
            num_units: decoder's num of cell units
        Returns:
            h_0: [batch_size, num_units]
            context_0: [batch_size, num_units]
        """
        h_0 = tf.zeros([1, self._num_units], self._dtype)
        context_0 = self._compute_context(h_0)
        h_0 = context_0 * 0

        if self._dec_init_states is None:
            batch_size = tf.shape(self._memory)[0]
            cell_states = self._cell.zero_state(batch_size, self._dtype)
        else:
            cell_states = self._dec_init_states

        attn_state_0 = AttnState(cell_states, h_0, context_0)

        return attn_state_0

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

    def __call__(self, inputs, attn_states):
        """
            inputs: emebeddings of previous word
            states: (cell_states, outputs) at each step
        """
        prev_cell_states, h, context = attn_states

        x = tf.concat([inputs, h, context], axis=-1)
        new_h, cell_states = self._cell.__call__(x, prev_cell_states)

        new_context = self._compute_context(new_h)
        new_attn_states = AttnState(cell_states, new_h, new_context)

        return (new_h, new_attn_states)
