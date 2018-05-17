# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-17 00:04:35
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-17 00:17:57

from tensorflow.contrib.rnn import LSTMStateTuple

import tensorflow as tf


def create_cell(num_units, cell_type, forget_bias=1.0):
    """
    cell: build a recurrent cell
        num_units: number of hidden cell units
        cell_type: LSTM, GRU, LN_LSTM
    """
    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=forget_bias)

    elif cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(num_units)

    elif cell_type == "LN_LSTM":
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)

    else:
        raise ValueError("Unknown cell type %s" % cell_type)

    return cell


def build_rnn_cell(num_layers, num_units, cell_type, forget_bias=1.0):
    """
    rnn_cell: build a multi-layer rnn cell
        num_layers: number of hidden layers
    """
    cell_seq = []
    for i in range(num_layers):
        cell = create_cell(num_units, cell_type, forget_bias)
        cell_seq.append(cell)

    if num_layers > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell_seq)
    else:
        rnn_cell = cell_seq[0]

    return rnn_cell


def self_attention_scores(hs, num_attn_hidden):
    """
    Pass hidden outputs to a feedforward net for self attention scores
    Use 2-layer DNN with tanh as in https://arxiv.org/pdf/1804.06658.pdf
        hs: [batch_size, max_time, num_hiddens]
    Returns:
        attn_scores: [batch_size, max_time, 1]
    """
    attn_a1 = tf.nn.tanh(tf.layers.dense(hs, num_attn_hidden))
    attn_a2 = tf.nn.tanh(tf.layers.dense(attn_a1, num_attn_hidden))
    e = tf.layers.dense(attn_a2, 1)
    attn_scores = tf.nn.softmax(tf.squeeze(e, axis=-1))
    attn_scores = tf.expand_dims(attn_scores, -1)

    return attn_scores


def build_emotion_regressor(
        embeddings, source_ids, num_layers, num_units, num_emotions,
        cell_type, forget_bias=1.0, bidir=False, self_attention=False,
        num_attn_hidden=128, dtype=tf.float32, name="encoder"):
    """
    Emotion Regressor for sentence following https://arxiv.org/abs/1804.06658
    Christos Baziotis et al., arXiv (2018), NTUA-SLP at SemEval-2018
        Task 1: Predicting Affective Content in Tweets
        with Deep Attentive RNNs and Transfer Learning

    encoder: build rnn encoder for Seq2seq
        source_ids: [batch_size, max_time]
        bidir: bidirectional or unidirectional

    Returns:
        encoder_outputs: [batch_size, max_time, num_units]
        encoder_states: (StateTuple(shape=(batch_size, num_units)), ...)
    """
    with tf.variable_scope(name):
        # embedding lookup, embed_inputs: [max_time, batch_size, num_units]
        embed_inputs = tf.nn.embedding_lookup(embeddings, source_ids)
        output_layer = tf.layers.Dense(num_emotions, name="output_layer")

        # bidirectional
        if bidir:
            encoder_states = []
            layer_inputs = embed_inputs

            for i in range(num_layers):
                with tf.variable_scope("layer_%d" % (i + 1)):
                    fw_cell = build_rnn_cell(
                        1, num_units, cell_type, forget_bias)
                    bw_cell = build_rnn_cell(
                        1, num_units, cell_type, forget_bias)

                    outs = tf.nn.bidirectional_dynamic_rnn(
                        fw_cell, bw_cell, layer_inputs,
                        dtype=dtype,
                        swap_memory=True)
                    bi_outputs, (state_fw, state_bw) = outs

                    if cell_type == "LSTM":
                        state_c = state_fw.c + state_bw.c
                        state_h = state_fw.h + state_bw.h
                        encoder_states.append(LSTMStateTuple(state_c, state_h))
                    else:
                        encoder_states.append(state_fw + state_bw)

                    if i < num_layers - 1:
                        layer_inputs = tf.layers.dense(
                            tf.concat(bi_outputs, -1), num_units)

            if self_attention:
                encoder_hs = tf.concat(bi_outputs, -1)
                attn_scores = self_attention_scores(
                    encoder_hs, num_attn_hidden)
                output_h = tf.reduce_sum(attn_scores * encoder_hs, axis=1)

            else:
                fw_h = bi_outputs[0][:, -1, :]
                bw_h = bi_outputs[1][:, 0, :]
                output_h = tf.concat([fw_h, bw_h], -1)

        # unidirectional
        else:
            rnn_cell = build_rnn_cell(
                num_layers, num_units, cell_type, forget_bias)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                rnn_cell, embed_inputs,
                dtype=dtype,
                swap_memory=True)

            if self_attention:
                attn_scores = self_attention_scores(
                    encoder_outputs, num_attn_hidden)
                output_h = tf.reduce_sum(attn_scores * encoder_hs, axis=-1)
            else:
                output_h = encoder_outputs[:, -1, :]

        outputs = tf.nn.sigmoid(output_layer(output_h))

    return outputs
