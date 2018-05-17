# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 23:54:40
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-16 19:28:18


from encoder import build_encoder
from decoder import build_decoder

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys


def init_embeddings(vocab_size, embed_size, dtype=tf.float32,
                    initializer=None, initial_values=None,
                    name='embeddings'):
    """
    embeddings:
        initialize trainable embeddings or load pretrained from files
    """
    with tf.variable_scope(name):
        if initial_values:
            embeddings = tf.Variable(initial_value=initial_values,
                                     name="embeddings", dtype=dtype)
        else:
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            embeddings = tf.Variable(
                initializer(shape=(vocab_size, embed_size)),
                name="embeddings", dtype=dtype)

        # id_0 represents SOS token, id_1 represents EOS token
        se_embed = tf.get_variable("SOS/EOS", [2, embed_size], dtype)
        # id_2 represents constant all zeros
        zero_embed = tf.zeros(shape=[1, embed_size])
        embeddings = tf.concat([se_embed, zero_embed, embeddings], axis=0)

    return embeddings


def compute_loss(source_ids, target_ids, sequence_mask, embeddings,
                 enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
                 dec_num_layers, dec_num_units, dec_cell_type, state_pass,
                 infer_batch_size, infer_type="greedy", beam_size=None,
                 max_iter=20, attn_wrapper=None, attn_num_units=128,
                 l2_regularize=None, name="Seq2seq"):
    """
    Creates a Seq2seq model and returns cross entropy loss.
    """
    with tf.name_scope(name):
        # build encoder
        encoder_outputs, encoder_states = build_encoder(
            embeddings, source_ids, enc_num_layers, enc_num_units,
            enc_cell_type, bidir=enc_bidir, name="%s_encoder" % name)

        # build decoder: logits, [batch_size, max_time, vocab_size]
        train_logits, infer_outputs = build_decoder(
            encoder_outputs, encoder_states, embeddings,
            dec_num_layers, dec_num_units, dec_cell_type,
            state_pass, infer_batch_size, attn_wrapper, attn_num_units,
            target_ids, infer_type, beam_size, max_iter,
            name="%s_decoder" % name)

        # compute loss
        with tf.name_scope('loss'):
            final_ids = tf.pad(target_ids, [[0, 0], [0, 1]], constant_values=1)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits, labels=final_ids)

            losses = tf.boolean_mask(losses, sequence_mask)
            reduced_loss = tf.reduce_mean(losses)
            CE = tf.reduce_sum(losses)

            if l2_regularize is None:
                return CE, reduced_loss, train_logits, infer_outputs
            else:
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                total_loss = reduced_loss + l2_regularize * l2_loss
                return CE, total_loss, train_logits, infer_outputs


def compute_perplexity(sess, CE, mask, feed_dict):
    """
    Compute perplexity for a batch of data
    """
    CE_words = sess.run(CE, feed_dict=feed_dict)
    N_words = np.sum(mask)
    return np.exp(CE_words / N_words)


def loadfile(filename, is_source, max_length):
    """
    Load and clean data
    """
    def clean(row):
        row = np.array(row.split(), dtype=np.int32)
        leng = len(row)
        if leng < max_length:
            if is_source:
                # represents constant zero padding
                pads = -np.ones(max_length - leng, dtype=np.int32)
                row = np.concatenate((pads, row))
            else:
                # represents EOS token
                pads = -2 * np.ones(max_length - leng, dtype=np.int32)
                row = np.concatenate((row, pads))
        elif leng > max_length:
            row = row[:max_length]
        return row

    df = pd.read_csv(filename, header=None, index_col=None)
    data = np.array(df[0].apply(lambda t: clean(t)).tolist(), dtype=np.int32)
    return data


def get_model_config(config):
    enc_num_layers = config["encoder"]["num_layers"]
    enc_num_units = config["encoder"]["num_units"]
    enc_cell_type = config["encoder"]["cell_type"]
    enc_bidir = config["encoder"]["bidirectional"]
    dec_num_layers = config["decoder"]["num_layers"]
    dec_num_units = config["decoder"]["num_units"]
    dec_cell_type = config["decoder"]["cell_type"]
    state_pass = config["decoder"]["state_pass"]
    infer_batch_size = config["inference"]["infer_batch_size"]
    infer_type = config["inference"]["type"]
    beam_size = config["inference"]["beam_size"]
    max_iter = config["inference"]["max_length"]
    attn_num_units = config["decoder"]["attn_num_units"]
    l2_regularize = config["training"]["l2_regularize"]

    return (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
            dec_num_layers, dec_num_units, dec_cell_type, state_pass,
            infer_batch_size, infer_type, beam_size, max_iter,
            attn_num_units, l2_regularize)


def get_training_config(config):
    train_config = config["training"]
    logdir = train_config["logdir"]
    restore_from = train_config["restore_from"]

    learning_rate = train_config["learning_rate"]
    gpu_fraction = train_config["gpu_fraction"]
    max_checkpoints = train_config["max_checkpoints"]
    train_steps = train_config["train_steps"]
    batch_size = train_config["batch_size"]
    print_every = train_config["print_every"]
    checkpoint_every = train_config["checkpoint_every"]

    s_filename = train_config["train_source_file"]
    t_filename = train_config["train_target_file"]
    s_max_leng = train_config["source_max_length"]
    t_max_leng = train_config["target_max_length"]

    dev_s_filename = train_config["dev_source_file"]
    dev_t_filename = train_config["dev_target_file"]

    loss_fig = train_config["loss_fig"]
    perp_fig = train_config["perplexity_fig"]

    return (logdir, restore_from, learning_rate, gpu_fraction, max_checkpoints,
            train_steps, batch_size, print_every, checkpoint_every,
            s_filename, t_filename, s_max_leng, t_max_leng, dev_s_filename,
            dev_t_filename, loss_fig, perp_fig)


def load(saver, sess, logdir):
    """
    Load the latest checkpoint
    """
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def save(saver, sess, logdir, step):
    """
    Save the checkpoint
    """
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')
