# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 21:37:34
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-14 21:41:21


from tensorflow.contrib.framework import nest

import tensorflow as tf


def transpose_batch_time(tensor):
    ndims = tensor.shape.ndims
    if ndims == 2:
        return tf.transpose(tensor, [1, 0])

    elif ndims == 3:
        return tf.transpose(tensor, [1, 0, 2])

    else:
        return tf.transpose(tensor, [1, 0, 2, 3])


# Dynamic decode function
def dynamic_decode(decoder_cell, max_iter):
    max_iter = tf.convert_to_tensor(max_iter, dtype=tf.int32)

    # TensorArray: wrap dynamic-sized, per-time-step, write-once Tensor arrays
    def create_tensor_array(d):
        # initial size = 0
        return tf.TensorArray(dtype=d, size=0, dynamic_size=True)

    time_index = tf.constant(0, dtype=tf.int32)
    # nest.map_structure: applies func to each entry in structure
    output_tensor_arrays = nest.map_structure(
        create_tensor_array, decoder_cell.output_dtype)

    cell_states, inputs, decode_finished = decoder_cell.initialize()

    # tf.while_loop(cond, body, vars): Repeat body while condition cond is true
    def condition(time_index, output_ta, cell_states, inputs, decode_finished):
        """
            if all "decode_finished" are True, return "False"
        """
        return tf.logical_not(tf.reduce_all(decode_finished))

    def body(time_index, output_ta, cell_states, inputs, decode_finished):
        sts = decoder_cell.step(time_index, cell_states, inputs,
                                decode_finished)
        new_output, new_cell_states, new_inputs, new_decode_finished = sts

        # TensorArray.write(index, value): register value and returns new TAs
        output_ta = nest.map_structure(
            lambda ta, out: ta.write(time_index, out),
            output_ta, new_output)

        new_decode_finished = tf.logical_or(
            tf.greater_equal(time_index, max_iter),
            new_decode_finished)

        return (time_index + 1, output_ta, new_cell_states, new_inputs,
                new_decode_finished)

    with tf.variable_scope("decoding"):

        res = tf.while_loop(
            condition,
            body,
            loop_vars=[time_index, output_tensor_arrays, cell_states,
                       inputs, decode_finished],
            back_prop=False)

    # get final outputs and states
    final_output_ta, final_cell_states = res[1], res[2]

    # TA.stack(): stack all tensors in TensorArray, [max_iter+1, batch_size, _]
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_output_ta)

    # finalize the computation from the decoder cell
    final_outputs = decoder_cell.finalize(final_outputs, final_cell_states)

    # transpose the final output
    final_outputs = nest.map_structure(transpose_batch_time, final_outputs)

    return final_outputs, final_cell_states
