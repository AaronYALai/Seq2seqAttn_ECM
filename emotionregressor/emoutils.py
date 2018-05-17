# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-17 00:00:36
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-17 00:16:19

from scipy.stats import pearsonr
from .emoregressor import build_emotion_regressor

import tensorflow as tf
import numpy as np
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

        # id_0 represents all-zero token, id_1 represents UNK token
        zero_embed = tf.zeros(shape=[1, embed_size])
        unk_embed = tf.get_variable("UNK", [1, embed_size], dtype)
        embeddings = tf.concat([zero_embed, unk_embed, embeddings], axis=0)

    return embeddings


def compute_loss(source_ids, targets, embeddings, num_layers, num_units,
                 num_emotions, cell_type, enc_bidir, self_attention=False,
                 num_attn_hidden=128, l2_regularize=None, name="emo_reg"):
    """
    Creates a emotion regressor and returns squared loss.
    """
    with tf.name_scope(name):
        # build emotion regressor
        outputs = build_emotion_regressor(
            embeddings, source_ids, num_layers, num_units,
            num_emotions, cell_type, bidir=enc_bidir,
            self_attention=self_attention, num_attn_hidden=num_attn_hidden,
            name=name)

        # compute loss
        with tf.name_scope('loss'):
            reduced_loss = tf.losses.mean_squared_error(
                labels=targets, predictions=outputs)

            if l2_regularize is None:
                return reduced_loss, outputs
            else:
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                total_loss = reduced_loss + l2_regularize * l2_loss
                return total_loss, outputs


def eval_mean_pearson(source_ids, predictions, sess, data, labels):
    """
    Compute Pearson's correlation coeff w.r.t. each emotion
    and average all coeffs as the evaluation metric.
    Ref: SemEval-2018 Task 1: Affect in Tweets (AIT-2018)
    """
    pred = sess.run(predictions, feed_dict={source_ids: data})
    pearsons = [pearsonr(pred[:, i], labels[:, i])[0] for i in range(4)]
    mean_pearson = np.mean(pearsons)
    return mean_pearson


def load(saver, sess, logdir):
    """
    Load the latest checkpoint
    Ref: https://github.com/ibab/tensorflow-wavenet
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
    Ref: https://github.com/ibab/tensorflow-wavenet
    """
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')
