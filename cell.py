# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 21:44:15
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-14 21:45:34


import tensorflow as tf


def create_cell(num_units, cell_type, forget_bias=1.0):
    """
    Cell: build a recurrent cell
        num_units: number of hidden cell units
        cell_type: LSTM, GRU, LN_LSTM (layer_normalize)
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
    RNN_cell: build a multi-layer rnn cell
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
