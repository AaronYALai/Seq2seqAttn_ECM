# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-17 00:18:12
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-18 01:52:52

from emoutils import init_embeddings, compute_loss, eval_mean_pearson, \
        loadfile, get_config, load, save

import argparse
import time
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    '''
    Parse the emotion regressor arguments.
    '''
    parser = argparse.ArgumentParser(
        description="Run emotion regressor training.")

    parser.add_argument('--config', nargs='?',
                        default='./config_emoBiLSTM_selfattn.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def main(args):
    # loading configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)["configuration"]

    name = config["Name"]

    # Load Data
    print("Loading data ...")
    max_length = config["training"]["max_length"]
    train_data, train_labels = loadfile("./emodata_train", max_length)
    dev_data, dev_labels = loadfile("./emodata_dev", max_length)
    word_ids = pd.read_csv("emodata_word_ids", header=None, sep="\t")

    # id_0 represents all-zero token
    embed_shift = 1
    train_data += embed_shift
    dev_data += embed_shift
    n_data = len(train_data)
    print("\tDone.")

    # Construct or load embeddings
    print("Initializing embeddings ...")
    vocab_size = len(word_ids)
    embed_size = config["embeddings"]["embed_size"]
    embeddings = init_embeddings(vocab_size, embed_size, name=name)
    print("\tDone.")

    (num_layers, num_units, num_emotions, cell_type, enc_bidir, self_attention,
     num_attn_hidden, logdir, restore_from, l2_regularize, learning_rate,
     gpu_fraction, max_checkpoints, train_steps, batch_size, print_every,
     checkpoint_every, loss_fig, pearson_fig) = get_config(config)

    # Build the model and compute losses
    source_ids = tf.placeholder(tf.int32, [None, None], name="source")
    targets = tf.placeholder(tf.float32, [None, num_emotions], name="target")

    print("Building model architecture ...")
    loss, predictions = compute_loss(
        source_ids, targets, embeddings, num_layers, num_units,
        num_emotions, cell_type, enc_bidir, self_attention,
        num_attn_hidden, l2_regularize, name)
    print("\tDone.")

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       epsilon=1e-4)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except Exception:
        print("Something went wrong while restoring checkpoint. "
              "Training is terminated to avoid the overwriting.")
        raise

    # ##### Training #####
    last_saved_step = saved_global_step
    num_steps = saved_global_step + train_steps
    losses = []
    dev_losses = []
    steps = []
    pearsons = []
    dev_pearsons = []

    print("Start training ...")
    try:
        for step in range(saved_global_step + 1, num_steps):
            start_time = time.time()

            indexes = np.random.choice(n_data, batch_size)
            batch_source = train_data[indexes]
            batch_labels = train_labels[indexes]

            feed_dict = {
                source_ids: batch_source,
                targets: batch_labels,
            }

            loss_value, _ = sess.run([loss, optim], feed_dict=feed_dict)
            dev_loss = sess.run(
                loss, feed_dict={source_ids: dev_data, targets: dev_labels})
            losses.append(loss_value)
            dev_losses.append(dev_loss)

            duration = time.time() - start_time
            if step % print_every == 0:
                train_mean_pearson = eval_mean_pearson(
                    source_ids, predictions, sess, batch_source, batch_labels)

                # evaluation on dev data
                dev_mean_pearson = eval_mean_pearson(
                    source_ids, predictions, sess, dev_data, dev_labels)

                steps.append(step)
                pearsons.append(train_mean_pearson)
                dev_pearsons.append(dev_mean_pearson)

                info = 'step {:d} - loss = {:.5f}, dev_loss = {:.5f}, '
                info += 'r = {:.4f}, dev_r = {:.4f}, ({:.3f} sec/step)'
                print(info.format(step, loss_value, dev_loss,
                      train_mean_pearson, dev_mean_pearson, duration))

            if step % checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C so save message is on its own line.
        print()

    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)

        # plot loss
        plt.figure()
        plt.plot(losses, label="train")
        plt.plot(dev_losses, label="dev")
        plt.title("squared loss")
        plt.xlabel("step")
        plt.legend()
        plt.savefig(loss_fig)

        # plot pearson's r
        plt.figure()
        plt.plot(steps, pearsons, label="train")
        plt.plot(steps, dev_pearsons, label="dev")
        plt.title("pearson's r")
        plt.xlabel("step")
        plt.legend()
        plt.savefig(pearson_fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
