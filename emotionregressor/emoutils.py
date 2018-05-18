# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-17 00:00:36
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-18 00:19:12

from scipy.stats import pearsonr
from emoregressor import build_emotion_regressor

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

        # id_0 represents all-zero token
        zero_embed = tf.zeros(shape=[1, embed_size])
        embeddings = tf.concat([zero_embed, embeddings], axis=0)

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


def loadfile(filename, max_length):
    df = pd.read_csv(filename, header=None)
    data = []
    labels = []
    for row in df[0].values:
        ids, emos = row.split(",")
        ids = np.array(ids.split(" "), dtype=np.int32)
        emos = np.array(emos.split(" "), dtype=np.float)

        if len(ids) < max_length:
            # represents constant zero padding
            pads = -np.ones(max_length - len(ids), dtype=np.int32)
            ids = np.concatenate((ids, pads))
        else:
            ids = ids[:max_length]

        data.append(ids)
        labels.append(emos)

    return np.array(data), np.array(labels)


def get_config(config):
    num_layers = config["model"]["num_layers"]
    num_units = config["model"]["num_units"]
    num_emotions = config["model"]["num_emotions"]
    cell_type = config["model"]["cell_type"]
    enc_bidir = config["model"]["bidirectional"]
    self_attention = config["model"]["self_attention"]
    num_attn_hidden = config["model"]["num_attn_hidden"]

    # infer_batch_size = config["inference"]["infer_batch_size"]
    # infer_type = config["inference"]["type"]
    # beam_size = config["inference"]["beam_size"]
    # max_iter = config["inference"]["max_length"]

    train_config = config["training"]
    logdir = train_config["logdir"]
    restore_from = train_config["restore_from"]
    l2_regularize = train_config["l2_regularize"]

    learning_rate = train_config["learning_rate"]
    gpu_fraction = train_config["gpu_fraction"]
    max_checkpoints = train_config["max_checkpoints"]
    train_steps = train_config["train_steps"]
    batch_size = train_config["batch_size"]
    print_every = train_config["print_every"]
    checkpoint_every = train_config["checkpoint_every"]

    loss_fig = train_config["loss_fig"]
    pearson_fig = train_config["pearson_fig"]

    return (num_layers, num_units, num_emotions, cell_type, enc_bidir,
            self_attention, num_attn_hidden, logdir, restore_from,
            l2_regularize, learning_rate, gpu_fraction, max_checkpoints,
            train_steps, batch_size, print_every, checkpoint_every,
            loss_fig, pearson_fig)


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
