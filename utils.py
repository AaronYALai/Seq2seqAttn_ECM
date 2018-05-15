# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 23:54:40
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-15 00:03:34


from encoder import build_encoder
from decoder import build_decoder

import tensorflow as tf
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
