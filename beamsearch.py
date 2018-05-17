# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 21:23:18
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-16 17:06:01


from tensorflow.contrib.framework import nest
from greedy import DecoderOutput

import tensorflow as tf
import numpy as np
import collections


class BeamDecoderOutput(collections.namedtuple(
        "BeamDecoderOutput", ("logits", "ids", "parents"))):
    """
        logits: [batch_size, beam_size, vocab_size]
        ids: [batch_size, beam_size], best words ids now
        parents: [batch_size, beam_size], previous step beam index ids
    """
    pass


class BeamDecoderCellStates(collections.namedtuple(
        "BeamDecoderCellStates", ("cell_states", "log_probs"))):
    """
        cell_states: [batch_size, beam_size, num_units]
        log_probs: [batch_size, beam_size]
    """
    pass


class BeamSearchDecodeCell(object):

    def __init__(self, embeddings, cell, dec_init_states,
                 output_layer, batch_size, dtype, beam_size,
                 vocab_size, div_gamma=None, div_prob=None):
        """
            div_gamma: (float) relative weight of penalties
            div_prob: (float) prob to apply penalties
        """
        self._embeddings = embeddings
        self._vocab_size = vocab_size
        self._cell = cell
        self._dec_init_states = dec_init_states
        self._output_layer = output_layer
        self._batch_size = batch_size
        self._start_token = tf.nn.embedding_lookup(embeddings, 0)
        self._end_id = 1
        self._dtype = dtype

        self._beam_size = beam_size
        self._div_gamma = div_gamma
        self._div_prob = div_prob
        if hasattr(self._cell, "_memory"):
            indices = np.repeat(np.arange(self._batch_size), self._beam_size)
            self._cell._memory = tf.gather(self._cell._memory, indices)

    @property
    def output_dtype(self):
        """Generate the structure for initial TensorArrays in dynamic_decode"""
        return BeamDecoderOutput(logits=self._dtype,
                                 ids=tf.int32, parents=tf.int32)

    def _initial_state(self):
        # t: [batch_size, num_units]
        cell_states = nest.map_structure(
            lambda t: tile_beam(t, self._beam_size), self._dec_init_states)

        # another "log_probs" initial states: accumulative log_prob!
        log_probs = tf.zeros([self._batch_size, self._beam_size],
                             dtype=self._dtype)

        return BeamDecoderCellStates(cell_states, log_probs)

    def initialize(self):
        # initial cell states
        cell_states = self._initial_state()

        # inputs: SOS, [embed_size] -> [batch_size, beam_size, embed_size]
        inputs = tf.tile(tf.reshape(self._start_token, [1, 1, -1]),
                         multiples=[self._batch_size, self._beam_size, 1])

        # initial ending signals: [batch_size, beam_size]
        decode_finished = tf.zeros([self._batch_size, self._beam_size],
                                   dtype=tf.bool)

        return cell_states, inputs, decode_finished

    def step(self, time_index, beam_states, inputs, decode_finished):
        """
            logits: [batch_size, beam_size, vocab_size]
            ids: [batch_size, beam_size], best words ids now
            parents: [batch_size, beam_size], previous step beam index ids
        """
        # 1-1: merge batch -> [batch_size*beam_size, ...]
        cell_states = nest.map_structure(
            merge_batch_beam, beam_states.cell_states)
        inputs = merge_batch_beam(inputs)

        # 1-2: perform cell ops to get new logits
        new_h, new_cell_states = self._cell.__call__(inputs, cell_states)
        logits = self._output_layer(new_h)

        # 1-3: split batch beam -> [batch_size, beam_size, ...]
        logits = split_batch_beam(logits, self._beam_size)
        new_cell_states = nest.map_structure(
            lambda t: split_batch_beam(t, self._beam_size), new_cell_states)

        # 2-1: compute log_probs, [batch_size, beam_size, vocab_size]
        step_log_probs = tf.nn.log_softmax(logits)
        step_log_probs = mask_log_probs(
            step_log_probs, self._end_id, decode_finished)

        # 2-2: add cumulative log_probs and "diversity penalty"
        log_probs = tf.expand_dims(beam_states.log_probs, axis=-1)
        log_probs = log_probs + step_log_probs
        log_probs = add_diversity_penalty(log_probs, self._div_gamma,
                                          self._div_prob, self._batch_size,
                                          self._beam_size, self._vocab_size)

        # 3-1: flatten, if time_index = 0, consider only one beam
        # log_probs[:, 0]: [batch_size, vocab_size]
        shape = [self._batch_size, self._beam_size * self._vocab_size]
        log_probs_flat = tf.reshape(log_probs, shape)
        log_probs_flat = tf.cond(time_index > 0, lambda: log_probs_flat,
                                 lambda: log_probs[:, 0])

        # 3-2: compute the top (beam_size) beams, [batch_size, beam_size]
        new_log_probs, indices = tf.nn.top_k(log_probs_flat, self._beam_size)

        # 3-3: obtain ids and parent beams
        new_ids = indices % self._vocab_size
        # //: floor division, know which beam it belongs to
        new_parents = indices // self._vocab_size

        # 4-1: compute new states
        new_inputs = tf.nn.embedding_lookup(self._embeddings, new_ids)

        decode_finished = gather_helper(
            decode_finished, new_parents, self._batch_size, self._beam_size)

        new_decode_finished = tf.logical_or(
            decode_finished, tf.equal(new_ids, self._end_id))

        new_cell_states = nest.map_structure(
            lambda t: gather_helper(t, new_parents, self._batch_size,
                                    self._beam_size), new_cell_states)

        # 4-2: create new state and output of decoder
        new_beam_states = BeamDecoderCellStates(cell_states=new_cell_states,
                                                log_probs=new_log_probs)
        new_output = BeamDecoderOutput(logits=logits, ids=new_ids,
                                       parents=new_parents)

        return (new_output, new_beam_states, new_inputs, new_decode_finished)

    def finalize(self, final_outputs, final_cell_states):
        """
            final_outputs: [max_time, logits] structure of tensor
            final_cell_states: BeamDecoderCellStates
        Returns:
            [max_time, batch_size, beam_size, ] stucture of tensor
        """
        # reverse the time dimension
        max_iter = tf.shape(final_outputs.ids)[0]
        final_outputs = nest.map_structure(lambda t: tf.reverse(t, axis=[0]),
                                           final_outputs)

        # initial states
        def create_ta(d):
            return tf.TensorArray(dtype=d, size=max_iter)

        f_time_index = tf.constant(0, dtype=tf.int32)
        # final output dtype
        final_dtype = DecoderOutput(logits=self._dtype, ids=tf.int32)
        f_output_ta = nest.map_structure(create_ta, final_dtype)

        # initial parents: [batch_size, beam_size]
        f_parents = tf.tile(
            tf.expand_dims(tf.range(self._beam_size), axis=0),
            multiples=[self._batch_size, 1])

        def condition(f_time_index, output_ta, f_parents):
            return tf.less(f_time_index, max_iter)

        def body(f_time_index, output_ta, f_parents):
            # get ids, logits and parents predicted at this time step
            input_t = nest.map_structure(lambda t: t[f_time_index],
                                         final_outputs)

            # parents: reversed version shows the next position to go
            new_beam_state = nest.map_structure(
                lambda t: gather_helper(t, f_parents, self._batch_size,
                                        self._beam_size),
                input_t)

            # create new output
            new_output = DecoderOutput(logits=new_beam_state.logits,
                                       ids=new_beam_state.ids)

            # write beam ids
            output_ta = nest.map_structure(
                lambda ta, out: ta.write(f_time_index, out),
                output_ta, new_output)

            return (f_time_index + 1), output_ta, input_t.parents

        with tf.variable_scope("beam_search_decoding"):
            res = tf.while_loop(
                    condition,
                    body,
                    loop_vars=[f_time_index, f_output_ta, f_parents],
                    back_prop=False)

        # stack the structure and reverse back
        final_outputs = nest.map_structure(lambda ta: ta.stack(), res[1])
        final_outputs = nest.map_structure(lambda t: tf.reverse(t, axis=[0]),
                                           final_outputs)

        return DecoderOutput(logits=final_outputs.logits,
                             ids=final_outputs.ids)


# ### Beam Search helpers ###
def tile_beam(tensor, beam_size):
    """
        tensor: batch-major, [batch_size, ...]
    Returns:
        tensor: beam_size tiled, [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    # set multiples: [1, beam_size, 1, ..., 1]
    multiples = [1 for i in range(tensor.shape.ndims)]
    multiples[1] = beam_size

    return tf.tile(tensor, multiples)


def merge_batch_beam(tensor):
    """
        tensor: [batch_size, beam_size, ...]
    Returns:
        tensorL [batch_size * beam_size, ...]
    """
    # tf.shape(t) handles indefinite shape
    batch_size = tf.shape(tensor)[0]
    # specified shape can be withdrawed right away
    beam_size = tensor.shape[1].value

    shape = list(tensor.shape)
    shape.pop(0)
    shape[0] = batch_size * beam_size

    return tf.reshape(tensor, shape)


def split_batch_beam(tensor, beam_size):
    """
        tensor: [batch_size * beam_size, ...]
    Returns:
        tensor: [batch_size, beam_size, ...]
    """
    shape = list(tensor.shape)
    shape[0] = beam_size
    shape.insert(0, -1)

    return tf.reshape(tensor, shape)


def mask_log_probs(log_probs, end_id, decode_finished):
    """
    Set log_probs after end_token to be [-inf, 0, -inf, ...]
        log_probs: [batch_size, beam_size, vocab_size]
        decode_finished: [batch_size, beam_size]
    """
    vocab_size = log_probs.shape[-1].value
    one_hot = tf.one_hot(end_id, vocab_size, on_value=0.0,
                         off_value=log_probs.dtype.min,
                         dtype=log_probs.dtype)
    I_fin = tf.expand_dims(tf.cast(decode_finished, log_probs.dtype),
                           axis=-1)

    return (1. - I_fin) * log_probs + I_fin * one_hot


def sample_bernoulli(prob, shape):
    """Samples a boolean tensor with shape = s according to bernouilli"""
    return tf.greater(prob, tf.random_uniform(shape))


def add_diversity_penalty(log_probs, div_gamma, div_prob, batch_size,
                          beam_size, vocab_size):
    """
    Diversity penalty by Li et al. 2016
        div_gamma: (float) diversity parameter
        div_prob: adds penalty with div_proba
    """
    if (div_gamma is None) or (div_prob is None):
        return log_probs

    if (div_gamma == 1) or (div_prob) == 0:
        return log_probs

    top_probs, top_inds = tf.nn.top_k(log_probs, k=vocab_size, sorted=True)

    # inverse permutation to get rank of each entry
    top_inds = tf.reshape(top_inds, [-1, vocab_size])
    index_rank = tf.map_fn(tf.invert_permutation, top_inds, back_prop=False)
    index_rank = tf.reshape(
        index_rank, shape=[batch_size, beam_size, vocab_size])

    # compute penalty
    penalties = tf.log(div_gamma) * tf.cast(index_rank, log_probs.dtype)

    # only apply penalty with some probability
    apply_penalty = tf.cast(
            sample_bernoulli(div_prob, [batch_size, beam_size, vocab_size]),
            penalties.dtype)
    penalties *= apply_penalty

    return log_probs + penalties


def gather_helper(tensor, indices, batch_size, beam_size):
    """
        tensor: [batch_size, beam_size, d]
        indices: [batch_size, beam_size]
    Returns:
        new_tensor: new_t[:, i] = t[:, new_parents[:, i]]
    """
    range_ = tf.expand_dims(tf.range(batch_size) * beam_size, axis=1)
    # flatten
    indices = tf.reshape(indices + range_, [-1])
    output = tf.gather(tf.reshape(tensor, [batch_size * beam_size, -1]),
                       indices)

    if tensor.shape.ndims == 2:
        return tf.reshape(output, [batch_size, beam_size])

    elif tensor.shape.ndims == 3:
        d = tensor.shape[-1].value
        return tf.reshape(output, [batch_size, beam_size, d])
