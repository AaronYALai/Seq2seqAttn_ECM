# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-18 00:18:10
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-18 14:41:44

from emoutils import init_embeddings, compute_loss, loadfile, get_config, load

import argparse
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd


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
    infer_filename = config["inference"]["infer_source_file"]
    max_length = config["inference"]["infer_source_max_length"]
    infer_data, _ = loadfile(infer_filename, max_length)
    word_ids = pd.read_csv("emodata_word_ids", header=None, sep="\t")

    # id_0 represents all-zero token
    embed_shift = 1
    infer_data += embed_shift
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
    _, predictions = compute_loss(
        source_ids, targets, embeddings, num_layers, num_units,
        num_emotions, cell_type, enc_bidir, self_attention,
        num_attn_hidden, l2_regularize, name)
    print("\tDone.")

    # Set up session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    try:
        saved_global_step = load(saver, sess, restore_from)
        if saved_global_step is None:
            raise ValueError("Cannot find the checkpoint to restore from.")

    except Exception:
        print("Something went wrong while restoring checkpoint. ")
        raise

    # ##### Inference #####
    print("Start inferring ...")
    emo_predicts = []
    infer_batch_size = config["inference"]["infer_batch_size"]
    n_data = infer_data.shape[0]
    n_pad = n_data % infer_batch_size
    if n_pad > 0:
        n_pad = infer_batch_size - n_pad

    pad = np.zeros((n_pad, infer_data.shape[1]), dtype=np.int32)
    infer_data = np.concatenate((infer_data, pad))

    for ith in range(int(len(infer_data) / infer_batch_size)):
        start = ith * infer_batch_size
        end = (ith + 1) * infer_batch_size
        batch = infer_data[start:end]

        preds = sess.run(predictions, feed_dict={source_ids: batch})
        emo_predicts.append(preds)

    emo_predicts = np.concatenate(emo_predicts)[:n_data]
    np.savetxt(config["inference"]["output_path"], emo_predicts, fmt='%.5f')
    print("\tDone.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
