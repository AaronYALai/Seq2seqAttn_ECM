# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 21:23:07
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-14 21:25:17


import tensorflow as tf
import collections


class DecoderOutput(collections.namedtuple(
                    "DecoderOutput", ("logits", "ids"))):
    """
        logits: [batch_size, vocab_size]
        ids: [batch_size]
    """
    pass


class GreedyDecodeCell(object):

    def __init__(self, embeddings, cell, dec_init_states,
                 output_layer, batch_size, dtype):
        self._embeddings = embeddings
        self._cell = cell
        self._dec_init_states = dec_init_states
        self._output_layer = output_layer
        self._batch_size = batch_size
        self._start_token = tf.nn.embedding_lookup(embeddings, 0)
        self._end_id = 1
        self._dtype = dtype

    @property
    def output_dtype(self):
        """Generate the structure for initial TensorArrays in dynamic_decode"""
        return DecoderOutput(logits=self._dtype, ids=tf.int32)

    def initialize(self):
        # initial cell states
        cell_states = self._dec_init_states

        # initial cell inputs: tile [1, embed_dim] => [batch_size, embed_dim]
        token = tf.expand_dims(self._start_token, 0)
        inputs = tf.tile(token, multiples=[self._batch_size, 1])

        # initial ending signals: start with all "False"
        decode_finished = tf.zeros(shape=[self._batch_size], dtype=tf.bool)

        return cell_states, inputs, decode_finished

    def step(self, time_index, cell_states, inputs, decode_finished):
        # next step of rnn cell and pass the output_layer for logits
        new_h, new_cell_states = self._cell.__call__(inputs, cell_states)
        logits = self._output_layer(new_h)

        # get ids of words predicted and their embeddings
        new_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        new_inputs = tf.nn.embedding_lookup(self._embeddings, new_ids)

        # make a new output for registering into TensorArrays
        new_output = DecoderOutput(logits, new_ids)

        # check whether the end_token is reached
        new_decode_finished = tf.logical_or(decode_finished,
                                            tf.equal(new_ids, self._end_id))

        return (new_output, new_cell_states, new_inputs, new_decode_finished)

    def finalize(self, final_outputs, final_cell_states):
        return final_outputs
