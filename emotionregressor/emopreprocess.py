# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-17 00:18:29
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-18 00:29:34

from nltk.tokenize import TweetTokenizer

import pandas as pd
import numpy as np


def loadfilename(filename):
    emotions = ["anger", "fear", "joy", "sadness"]
    data = {}

    for ith, emo in enumerate(emotions):
        df = pd.read_csv(filename.format(emo), "\t").values
        for ID, tweet, _, score in df:
            if ID not in data:
                # emotion is a 4-dimensional real-valued vector
                data[ID] = {
                    "text": tweet,
                    "emotion": np.zeros(4)
                }

            data[ID]["emotion"][ith] = float(score)

    return data


def tokenize(data):
    """
    NLTK Tweet: Tokenizations for training data, build a dictionary
        data: {ID: {text: tweet, emotion: vector}}
    """
    twtknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    words = {}
    index = 0
    tk_r = []
    strings = []

    # Build dictionary
    for ID in data.keys():
        tk_r.append(twtknzr.tokenize(data[ID]["text"]))
        tk_ids = []

        for word in tk_r[-1]:
            # strip hashtag's #
            if word[0] == "#":
                word = word[1:]

            if word not in words:
                words[word] = index
                index += 1

            strings.append(word)
            tk_ids.append(words[word])

        # transform text into token_id sequence
        data[ID]["ids"] = tk_ids

    return words


def tokenize_test(data, words):
    """
    NLTK Tweet: Tokenizations for dev/valid/test data
        data: {ID: {text: tweet, emotion: vector}}
        words: predefined dictionary
    """
    twtknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    num_unk = 0
    total_word = 0
    tk_r = []

    for ID in data.keys():
        tk_r.append(twtknzr.tokenize(data[ID]["text"]))
        tk_ids = []
        for word in tk_r[-1]:
            if word[0] == "#":
                word = word[1:]

            if word not in words:
                num_unk += 1
            else:
                tk_ids.append(words[word])

            total_word += 1

        data[ID]["ids"] = tk_ids

    return num_unk, total_word


def to_output_form(data):
    """
    Transform data into a list of strings output format
    """
    processed = []
    for ID in data.keys():
        text = " ".join(np.array(data[ID]["ids"], dtype=str).tolist())
        emo = " ".join(data[ID]["emotion"].astype(str).tolist())
        processed.append(text + "," + emo)

    return processed


def preprocess():
    train_filename = "./EI-reg-En-train/EI-reg-En-{}-train.txt"
    train_data = loadfilename(train_filename)
    train_words = tokenize(train_data)
    train_processed = to_output_form(train_data)

    train_df = pd.DataFrame(data={"0": train_processed})
    train_df.to_csv("emodata_train", header=None, index=None)

    word_df = pd.DataFrame(data={"0": train_words})
    word_df.to_csv("emodata_word_ids", header=None, sep="\t")
    print("Training set processed. With {} words and {} sentences.".format(
          len(train_words), len(train_data)))

    dev_filename = "./2018-EI-reg-En-dev/2018-EI-reg-En-{}-dev.txt"
    dev_data = loadfilename(dev_filename)
    num_unk, total_word = tokenize_test(dev_data, train_words)
    dev_processed = to_output_form(dev_data)

    dev_df = pd.DataFrame(data={"0": dev_processed})
    dev_df.to_csv("emodata_dev", header=None, index=None)
    print("Dev set processed: {} UNKs, {} words and {} sentences.".format(
          num_unk, total_word, len(dev_data)))


if __name__ == "__main__":
    preprocess()
