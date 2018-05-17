# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 19:03:01
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-16 22:27:43


from tensorflow.contrib.rnn import LSTMStateTuple
from .cell import build_rnn_cell

import tensorflow as tf


def build_encoder(embeddings, source_ids, num_layers, num_units, cell_type,
                  forget_bias=1.0, bidir=False, time_major=False,
                  dtype=tf.float32, name="encoder"):
    """
    encoder: build rnn encoder for Seq2seq
        source_ids: [batch_size, max_time]
        bidir: bidirectional or unidirectional

    Returns:
        encoder_outputs: [batch_size, max_time, num_units]
        encoder_states: (StateTuple(shape=(batch_size, num_units)), ...)
    """
    with tf.variable_scope(name):
        if time_major:
            source_ids = tf.transpose(source_ids)

        # embedding lookup, embed_inputs: [max_time, batch_size, num_units]
        embed_inputs = tf.nn.embedding_lookup(embeddings, source_ids)

        # bidirectional
        if bidir:
            encoder_states = []
            layer_inputs = embed_inputs

            # build rnn layer-by-layer
            for i in range(num_layers):
                with tf.variable_scope("layer_%d" % (i + 1)):
                    fw_cell = build_rnn_cell(
                        1, num_units, cell_type, forget_bias)
                    bw_cell = build_rnn_cell(
                        1, num_units, cell_type, forget_bias)

                    dyn_rnn = tf.nn.bidirectional_dynamic_rnn(
                        fw_cell, bw_cell, layer_inputs,
                        time_major=time_major,
                        dtype=dtype,
                        swap_memory=True)
                    bi_outputs, (state_fw, state_bw) = dyn_rnn

                    # handle cell memory state
                    if cell_type == "LSTM":
                        state_c = state_fw.c + state_bw.c
                        state_h = state_fw.h + state_bw.h
                        encoder_states.append(LSTMStateTuple(state_c, state_h))
                    else:
                        encoder_states.append(state_fw + state_bw)

                    # concat and map as inputs of next layer
                    layer_inputs = tf.layers.dense(
                        tf.concat(bi_outputs, -1), num_units)

            encoder_outputs = layer_inputs
            encoder_states = tuple(encoder_states)

        # unidirectional
        else:
            rnn_cell = build_rnn_cell(
                num_layers, num_units, cell_type, forget_bias)

            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                rnn_cell, embed_inputs,
                time_major=time_major,
                dtype=dtype,
                swap_memory=True)

    return encoder_outputs, encoder_states
