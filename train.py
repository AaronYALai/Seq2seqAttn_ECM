# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-14 19:08:20
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-16 16:50:44


from utils import init_embeddings, compute_loss, compute_perplexity, \
        loadfile, get_model_config, get_training_config, load, save
from attention import AttentionWrapper
from ECM import ECMWrapper

import argparse
import time
import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    '''
    Parse Seq2seq with attention/ECM arguments.
    '''
    parser = argparse.ArgumentParser(description="Run seq2seq training.")

    parser.add_argument('--configuration', nargs='?',
                        default='./config_seq2seqAttn_beamsearch.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def main():
    # loading configurations
    with open(args.configuration) as f:
        config = yaml.safe_load(f)["configuration"]

    name = config["Name"]

    # Construct or load embeddings
    vocab_size = config["embeddings"]["vocab_size"]
    embed_size = config["embeddings"]["embed_size"]
    embeddings = init_embeddings(vocab_size, embed_size)

    # Build the model and compute losses
    source_ids = tf.placeholder(tf.int32, [None, None], name="source")
    target_ids = tf.placeholder(tf.int32, [None, None], name="target")
    sequence_mask = tf.placeholder(tf.bool, [None, None], name="mask")

    attn_wrappers = {
        "None": None,
        "Attention": AttentionWrapper,
        "ECM": ECMWrapper,
    }
    attn_wrapper = attn_wrappers.get(config["decoder"]["wrapper"])

    (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
     dec_num_layers, dec_num_units, dec_cell_type, state_pass,
     infer_batch_size, infer_type, beam_size, max_iter,
     attn_num_units, l2_regularize) = get_model_config(config)

    CE, loss, logits, infer_outputs = compute_loss(
        source_ids, target_ids, sequence_mask, embeddings,
        enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
        dec_num_layers, dec_num_units, dec_cell_type, state_pass,
        infer_batch_size, infer_type, beam_size, max_iter,
        attn_wrapper, attn_num_units, l2_regularize, name)

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    (logdir, restore_from, learning_rate, gpu_fraction, max_checkpoints,
     train_steps, batch_size, print_every, s_filename, t_filename,
     s_max_leng, t_max_leng, dev_s_filename, dev_t_filename,
     loss_fig, perp_fig) = get_training_config(config)

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

    # Training the model
    last_saved_step = saved_global_step
    num_steps = saved_global_step + train_steps
    losses = []
    steps = []
    perps = []
    dev_perps = []
    # id_0, id_1, id_2 preserved for SOS, EOS, constant zero padding
    embed_shift = 3

    source_data = loadfile(s_filename, is_source=True,
                           max_length=s_max_leng) + embed_shift
    target_data = loadfile(t_filename, is_source=False,
                           max_length=t_max_leng) + embed_shift
    masks = (target_data != -1)
    n_data = len(source_data)

    dev_source_data = None
    if dev_s_filename:
        dev_source_data = loadfile(dev_s_filename, is_source=True,
                                   max_length=s_max_leng) + embed_shift
        dev_target_data = loadfile(dev_t_filename, is_source=False,
                                   max_length=t_max_leng) + embed_shift
        dev_masks = (dev_target_data != -1)

    try:
        for step in range(saved_global_step + 1, num_steps):
            start_time = time.time()
            rand_indexes = np.random.choice(n_data, batch_size)
            source_batch = source_data[rand_indexes]
            target_batch = target_data[rand_indexes]
            mask_batch = masks[rand_indexes]

            feed_dict = {
                source_ids: source_batch,
                target_ids: target_batch,
                sequence_mask: mask_batch,
            }

            loss_value, _ = sess.run([loss, optim], feed_dict=feed_dict)
            losses.append(loss_value)

            duration = time.time() - start_time

            if step % print_every == 0:
                # train perplexity
                t_perp = compute_perplexity(sess, CE, mask_batch, feed_dict)
                perps.append(t_perp)

                # dev perplexity
                dev_str = ""
                if dev_source_data:
                    dev_feed_dict = {
                        source_ids: dev_source_data,
                        target_ids: dev_target_data,
                        sequence_mask: dev_masks,
                    }
                    dev_perp = compute_perplexity(
                        sess, CE, dev_masks, dev_feed_dict)
                    dev_perps.append(dev_perp)
                    dev_str = "dev_prep: {:.3f}, ".format(dev_perp)

                steps.append(step)
                info = 'step {:d}, loss = {:.6f}, '
                info += 'perp: {:.3f}, {}({:.3f} sec/step)'
                print(info.format(step, loss_value, t_perp, dev_str, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C so save message is on its own line.
        print()

    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)

        plt.figure()
        plt.plot(losses)
        plt.savefig(loss_fig)

        plt.figure()
        plt.plot(steps, perps)
        if dev_source_data:
            plt.plot(steps, dev_perps)
        plt.savefig(perp_fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
