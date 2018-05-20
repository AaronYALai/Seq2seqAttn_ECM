# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2018-05-15 00:08:10
# @Last Modified by:   AaronLai
# @Last Modified time: 2018-05-20 13:41:46

import pandas as pd
import time
import numpy as np

from nltk.tokenize import TweetTokenizer


def sentence_to_ids(sentence, dictionary):
    """
    Transform sentence into ids and record new words
        sentence: [word_0, word_1, ...]
        dictionary: {word: [id, frequency]}
    """
    ids = []
    for i in range(len(sentence)):
        # strip hashtag's #
        if sentence[i][0] == "#":
            sentence[i] = sentence[i][1:]

        if sentence[i] not in dictionary:
            dictionary[sentence[i]] = [len(dictionary), 0]

        # count word frequency
        dictionary[sentence[i]][1] += 1

        ids.append(dictionary[sentence[i]][0])

    return sentence, ids


def tokenize_and_build_dictionary(sentences, print_every=100000):
    """
    NLTK Tweet: Tokenizations for training data, build a dictionary
        sentences: [sentence_0, sentence_1, ...]
    Returns:
        data: list of {message:[...], m_ids:[...], response:[...], r_ids:[...]}
        dictionary: {word: [id, frequency]}
    """
    twtknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    data = []
    dictionary = {}

    # Build dictionary
    st = time.time()
    for i in range(0, len(sentences), 2):
        pair = {}
        message = twtknzr.tokenize(sentences[i])
        response = twtknzr.tokenize(sentences[i + 1])

        # transform text into token_id sequence
        pair["message"], pair["m_ids"] = sentence_to_ids(message, dictionary)
        pair["response"], pair["r_ids"] = sentence_to_ids(response, dictionary)

        if i % print_every == 0:
            print("Processed {} sentences, used {:.4f} seconds.".format(
                  i, time.time() - st))

        data.append(pair)

    return data, dictionary


def process_sentence(sentence, dictionary):
    """
    Process sentence with the new dictionary
        1. Drop words not in the dictionary
        2. Return a new list of word ids
    """
    processed_sent = []
    processed_ids = []
    for word in sentence:
        if dictionary.get(word) is not None:
            processed_sent.append(word)
            processed_ids.append(dictionary[word])

    return processed_sent, processed_ids


def to_output_form(token_list):
    """
    Transform data into a string output format
    """
    output = " ".join(np.array(token_list, dtype=str).tolist())

    return output


def export_data(data):
    """
    Save data as raw text files for messages, responses and their ids
    """
    df = pd.DataFrame(data).applymap(to_output_form)

    df["message"].to_csv("twitter_message.txt", header=None, index=None)
    df["response"].to_csv("twitter_response.txt", header=None, index=None)

    df["m_ids"].to_csv("twitter_message_ids.txt", header=None, index=None)
    df["r_ids"].to_csv("twitter_response_ids.txt", header=None, index=None)


def preprocess():
    # Load data: availabel from https://github.com/Marsan-Ma/chat_corpus
    df = pd.read_csv("twitter_en_big.txt", header=None, sep="\t")
    data, dictionary = tokenize_and_build_dictionary(df[0].tolist())

    total_words = sum([len(d["m_ids"]) + len(d["r_ids"]) for d in data])
    dict_size = len(dictionary)

    print("Total words: {}; Vocabulary size: {}".format(
          total_words, dict_size))

    # Sort dictionary by word frequency
    word_freq = [(dictionary[key][1], key) for key in dictionary.keys()]
    words_sort_by_freq = sorted(word_freq, reverse=True)
    freq = np.asarray(words_sort_by_freq)[:, 0].astype(np.int32)

    # Set the vocabulary size threshold
    num_keep_words = 50000

    # Number of words that will be dropped
    num_words_drop = freq[num_keep_words:].sum()
    percent_drop = num_words_drop / total_words

    info = "Keep the most frequent {} words will drop {} tokens "
    info += "which is {:.3f} percent."
    print(info.format(num_keep_words, num_words_drop, 100 * percent_drop))

    # Build new dictionary
    dictionary = {}
    for index, freq_word in enumerate(words_sort_by_freq[:num_keep_words]):
        dictionary[freq_word[1]] = index

    for i in range(len(data)):
        data[i]["message"], data[i]["m_ids"] = process_sentence(
            data[i]["message"], dictionary)

        data[i]["response"], data[i]["r_ids"] = process_sentence(
            data[i]["response"], dictionary)

    p_total_words = sum([len(d["m_ids"]) + len(d["r_ids"]) for d in data])
    p_dict_size = len(dictionary)

    print("Processed total words: {}; New vocabulary size: {}".format(
          p_total_words, p_dict_size))

    # Export data
    export_data(data)

    print("Done.")


if __name__ == "__main__":
    preprocess()
