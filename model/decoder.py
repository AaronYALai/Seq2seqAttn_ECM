# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 19:07:14
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-25 18:14:10


from .cell import build_rnn_cell
from .greedy import GreedyDecodeCell
from .beamsearch import BeamSearchDecodeCell, ECMBeamSearchDecodeCell
from .dymdecode import dynamic_decode
from .ECM import ECMWrapper

import tensorflow as tf
import warnings


def build_decoder(encoder_outputs, encoder_states, embeddings,
                  num_layers, num_units, cell_type,
                  state_pass=True, infer_batch_size=None,
                  attention_wrap=None, attn_num_units=128,
                  target_ids=None, infer_type="greedy", beam_size=None,
                  max_iter=20, dtype=tf.float32, forget_bias=1.0,
                  name="decoder"):
    """
    decoder: build rnn decoder with attention and
        target_ids: [batch_size, max_time]
        infer_type: greedy decode or beam search
        attention_wrap: a wrapper to enable attention mechanism

    Returns:
        train_outputs: logits, [batch_size, max_time, vocab_size]
        infer_outputs: namedtuple(logits, ids), [batch_size, max_time, d]
    """
    # parameter checking
    if infer_batch_size is None:
        txt = "infer_batch_size not specified, infer output will be 'None'."
        warnings.warn(txt)
    elif infer_type == "beam_search" and beam_size is None:
        raise ValueError("Inference by beam search must specify beam_size.")

    if target_ids is None:
        txt = "target_ids not specified, train_outputs will be 'None'."
        warnings.warn(txt)

    # Build decoder
    with tf.variable_scope(name):
        vocab_size = embeddings.shape[0].value

        # decoder rnn_cell
        cell = build_rnn_cell(num_layers, num_units, cell_type, forget_bias)
        dec_init_states = encoder_states if state_pass else None
        output_layer = tf.layers.Dense(
            vocab_size, use_bias=False, name="output_projection")

        # wrap with attention
        if attention_wrap is not None:
            memory = encoder_outputs
            cell = attention_wrap(
                cell, memory, dec_init_states, attn_num_units,
                num_units, dtype)

            dec_init_states = cell.initial_state()

        # Decode - for training
        # pad the token sequences with SOS (Start of Sentence)
        train_outputs = None
        if target_ids is not None:
            input_ids = tf.pad(target_ids, [[0, 0], [1, 0]], constant_values=0)
            embed_inputs = tf.nn.embedding_lookup(embeddings, input_ids)

            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                cell, embed_inputs,
                initial_state=dec_init_states,
                dtype=dtype,
                swap_memory=True)

            # logits
            train_outputs = output_layer(decoder_outputs)

        # Decode - for inference
        infer_outputs = None
        if infer_batch_size is not None:
            if dec_init_states is None:
                dec_init_states = cell.zero_state(infer_batch_size, dtype)

            if infer_type == "beam_search":
                decoder_cell = BeamSearchDecodeCell(
                    embeddings, cell, dec_init_states, output_layer,
                    infer_batch_size, dtype, beam_size, vocab_size,
                    div_gamma=None, div_prob=None)

            else:
                decoder_cell = GreedyDecodeCell(
                    embeddings, cell, dec_init_states, output_layer,
                    infer_batch_size, dtype)

            # namedtuple(logits, ids)
            infer_outputs, _ = dynamic_decode(decoder_cell, max_iter)

    return train_outputs, infer_outputs


def build_ECM_decoder(encoder_outputs, encoder_states, embeddings, num_layers,
                      num_units, cell_type, num_emo, emo_cat, emo_cat_units,
                      emo_int_units, state_pass=True, infer_batch_size=None,
                      attn_num_units=128, target_ids=None, beam_size=None,
                      max_iter=20, dtype=tf.float32, forget_bias=1.0,
                      name="ECM_decoder"):
    """
    ECM decoder: build ECM decoder with emotion category embedding,
             internal & external memory modules
        target_ids: [batch_size, max_time]
        infer_type: greedy decode or beam search
        attention_wrap: a wrapper to enable attention mechanism
        num_emo: number of emotions
        emo_cat: emotion catogories, [batch_size]
        emo_cat_units: dimension of emotion category embeddings
        emo_int_units: dimension of emotion internal memory

    Returns:
        train_outputs: (generic_logits, emo_ext_logits, alphas, int_M_emo),
            first 3 shape: [batch_size, max_time, d]
            int_M_emo: [batch_size, emo_int_units]
        infer_outputs: namedtuple(logits, ids), [batch_size, max_time, d]
    """
    # parameter checking
    if infer_batch_size is None:
        txt = "infer_batch_size not specified, infer output will be 'None'."
        warnings.warn(txt)
    elif beam_size is None:
        raise ValueError("Inference by beam search must specify beam_size.")

    if target_ids is None:
        txt = "target_ids not specified, train_outputs will be 'None'."
        warnings.warn(txt)

    with tf.variable_scope(name):
        vocab_size = embeddings.shape[0].value

        # create emotion category embeddings
        emo_init = tf.contrib.layers.xavier_initializer()
        emo_cat_embeddings = tf.Variable(
            emo_init(shape=(num_emo, emo_cat_units)),
            name="emo_cat_embeddings", dtype=dtype)
        emo_cat_embs = tf.nn.embedding_lookup(emo_cat_embeddings, emo_cat)

        # decoder rnn_cell
        cell = build_rnn_cell(num_layers, num_units, cell_type, forget_bias)
        dec_init_states = encoder_states if state_pass else None
        output_layer = tf.layers.Dense(
            vocab_size, use_bias=False, name="output_projection")

        # wrap with ECM internal memory module
        memory = encoder_outputs

        cell = ECMWrapper(
            cell, memory, dec_init_states, attn_num_units, num_units, dtype,
            emo_cat_embs, emo_cat, num_emo, emo_int_units)

        dec_init_states = cell.initial_state()

        # ECM external memory module
        emo_output_layer = tf.layers.Dense(
            vocab_size, use_bias=False, name="emo_output_projection")

        emo_choice_layer = tf.layers.Dense(
            1, use_bias=False, name="emo_choice_alpha")

        # Decode - for training
        # pad the token sequences with SOS (Start of Sentence)
        train_outputs = None
        if target_ids is not None:
            input_ids = tf.pad(target_ids, [[0, 0], [1, 0]], constant_values=0)
            embed_inputs = tf.nn.embedding_lookup(embeddings, input_ids)

            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                cell, embed_inputs,
                initial_state=dec_init_states,
                dtype=dtype,
                swap_memory=True)

            # logits & final internal memory states
            generic_logits = output_layer(decoder_outputs)
            emo_ext_logits = emo_output_layer(decoder_outputs)
            alphas = tf.nn.sigmoid(emo_choice_layer(decoder_outputs))
            int_M_emo = decoder_states.internal_memory

            train_outputs = (generic_logits, emo_ext_logits, alphas, int_M_emo)

        # Decode - for inference, beam search
        infer_outputs = None
        if infer_batch_size is not None:
            if dec_init_states is None:
                dec_init_states = cell.zero_state(infer_batch_size, dtype)

            decoder_cell = ECMBeamSearchDecodeCell(
                embeddings, cell, dec_init_states, output_layer,
                emo_output_layer, emo_choice_layer,
                infer_batch_size, dtype, beam_size, vocab_size,
                div_gamma=None, div_prob=None)

            # namedtuple(logits, ids)
            infer_outputs, _ = dynamic_decode(decoder_cell, max_iter)

    return cell, train_outputs, infer_outputs
