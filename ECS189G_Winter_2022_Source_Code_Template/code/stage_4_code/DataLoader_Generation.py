#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
from code.base_class.dataset import dataset
import torch
import numpy as np
import pickle
import pandas as pd
from collections import Counter
import itertools

class Dataset_Loader(dataset):
    dataset_name = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_w_to_ind = None
    vocab_ind_to_w = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_raw(self):

        if self.dataset_source_folder_path is not None:
            path = self.dataset_source_folder_path
        else:
            path = "/Users/jacquelinemitchell/Documents/ECS189G/sample_code/ECS189G-21" \
               "W/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/data"

        # The longest length of any joke is 204 chars, so we'll make our sequence length this.
        jokes = []
        jokes_lengths = [] # jokes lenghts will be useful for inference later, but not immediatlye useful
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue # skip this is the header
                line = line.split(",", 1)[1].replace("\n", "").replace("\"", "").lower()
                line = line.split()
                jokes.append(line)
                jokes_lengths.append(len(line))
        return jokes, jokes_lengths

    def create_vocabulary(self, cleaned_strings):

        all_words = list(itertools.chain.from_iterable(cleaned_strings))
        counts = Counter(all_words)

        vocabulary_w_to_i = {} # word to index
        vocabulary_i_to_w = {} # index to word
        for i, item in enumerate(counts.most_common()):
            word = item[0]
            vocabulary_w_to_i[word] = i
            vocabulary_i_to_w[i] = word

        self.vocab_w_to_ind = vocabulary_w_to_i
        self.vocab_ind_to_w = vocabulary_i_to_w

        return vocabulary_w_to_i, vocabulary_i_to_w

    def prepare_input_sequences(self, sequences, vocab, sequence_length=3):
        # note: vocab is the word to index one.

        entire_corpus = list(itertools.chain.from_iterable(sequences))
        encoded_corpus = [vocab[i] for i in entire_corpus]

        previous_3 = [] # X's
        next_3 = []  # y's

        for i in range(0, len(encoded_corpus) - sequence_length, 1):
            prev = encoded_corpus[i:i+sequence_length]
            #next = encoded_corpus[i+sequence_length:i+sequence_length+sequence_length]
            next = encoded_corpus[i + 1:i + sequence_length + 1]

            if len(prev) != sequence_length or len(next) != sequence_length:
                assert True == False

            previous_3.append(prev)
            next_3.append(next)

        return previous_3, next_3

    def load(self):
        jokes, jokes_len = self.load_raw()
        vocab_w_i, vocab_i_w = self.create_vocabulary(jokes)
        context, targets = self.prepare_input_sequences(jokes, vocab_w_i)
        return context, targets, vocab_w_i, vocab_i_w, jokes, jokes_len


if __name__ == "__main__":
    a = Dataset_Loader()
    j = a.load_raw()
    print(j)
    print(["ji", "b"])
    c, d = a.create_vocabulary(j)
    p3, n3 = a.prepare_input_sequences(j, c)

    j = 0
    for i in zip(p3, n3):
        print(i)
        a = list(itertools.chain.from_iterable(i))
        print([d[j] for j in a])
        if j == 2:
            break
        j += 1
    print(len(p3))




