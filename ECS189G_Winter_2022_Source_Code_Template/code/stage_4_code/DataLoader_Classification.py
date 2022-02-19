from code.base_class.dataset import dataset
import os, re, itertools
import numpy as np
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import random

random.seed(2)
np.random.seed(2)

class Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_raw_data(self):

        parent_path = "/Users/jacquelinemitchell/Documents/ECS189G/sample_code/ECS189G" \
                      "-21W/ECS189G_Winter_2022_Source_Code_Template/" \
                      "data/stage_4_data/text_classification/"
        data = {}
        labels = {}

        # Load the training examples and testing examples
        for group in ["train", "test"]:
            reviews = []
            review_labels = []
            for rating in ['pos', 'neg']:
                path = parent_path + f"/{group}/{rating}"
                for file in os.listdir(path):
                    if os.fsdecode(file).endswith(".txt"):
                        with open(path + "/" + file) as text_file:
                            text = text_file.read()
                            label = {"pos" : 1, "neg" : 0}[rating]
                            reviews.append(text)
                            review_labels.append(label)
                data[group] = np.array(reviews)
                labels[group] = np.array(review_labels)

        return data, labels

    def process_raw(self, data, labels):

        # We need to shuffle so that we don't have the same kind of label happening after
        # eachother over and over again...
        X_train_raw, y_train = shuffle(data["train"], labels["train"])
        X_test_raw, y_test = shuffle(data["test"], labels["test"])

        # Training Data
        X_train, X_test = [], []

        # a = 0
        b = 100

        # Now, let us preprocess the data, upon inspection the data has html tags, so
        porter = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        for i, data in enumerate(X_train_raw[:b]):
            #print(i)
            # We want to get rid of any ",", "'", """; these aren't words
            cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", BeautifulSoup(data, "html.parser").get_text().lower()).split()
            # We will now remove common words like "and", "or", "but" (stop words)

            cleaned_text = [w for w in cleaned_text if w not in stop_words]
            stemmed = [porter.stem(word) for word in cleaned_text]
            X_train.append(stemmed)
            #print(X_train[i])

           # if i == 0:
               # print(X_train[0])

        for i, data in enumerate(X_test_raw[:b]):
            cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", BeautifulSoup(data, "html.parser").get_text().lower()).split()
            # We will now remove common words like "and", "or", "but" (stop words)

            cleaned_text = [w for w in cleaned_text if w not in stop_words]
            stemmed = [porter.stem(word) for word in cleaned_text]
            X_test.append(stemmed)
           # print(X_test[i])

        return {"X_train" : X_train, "y_train" : y_train[:b], "X_test" : X_test, "y_test" : y_test[:b]}

    def construct_vocab(self, data_dict, dict_size=5000):

        print("Counting")
        # We are going to use the collection items and iterate throughout the keys, in decreasing order
        all_words = list(itertools.chain.from_iterable(data_dict["X_train"]))
        counts = Counter(all_words)
        # print(len(counts))
        # print(counts)

        # Note: the reason why we have this is because we want to use
        # index 0 to be the stop word and index 1 to be the out-of dictionary value
        vocabulary = {}
        for i, item in enumerate(counts.most_common(dict_size-2)):
            key = item[0]
            vocabulary[key] = i + 2

        # print(vocabulary)
        # print(len(vocabulary))
        return vocabulary

    # Pad and prep inspired by https://github.com/Supearnesh/ml-imdb-rnn#step-1---data-collection
    def pad_and_prep(self, vocabulary, data_dict, max_sentence_length=200):
        dummy_pad = 0
        unknown_word = 1

        all_sentences_train = []
        all_sentences_test = []

        for sentence in data_dict["X_train"]:
            #print(sentence)
            padded_sentence = [dummy_pad] * max_sentence_length
            for i, word in enumerate(sentence[:max_sentence_length]):
                if word in vocabulary:
                    padded_sentence[i] = vocabulary[word]
                else:
                    # This word isn't known in our vocab, so we have to make this known to the model
                    padded_sentence[i] = unknown_word
            all_sentences_train.append(padded_sentence)

        for sentence in data_dict["X_test"]:
            #print(sentence)
            padded_sentence = [dummy_pad] * max_sentence_length
            for i, word in enumerate(sentence[:max_sentence_length]):
                if word in vocabulary:
                    padded_sentence[i] = vocabulary[word]
                else:
                    # This word isn't known in our vocab, so we have to make this known to the model
                    padded_sentence[i] = unknown_word
            all_sentences_test.append(padded_sentence)

        return np.array(all_sentences_train), np.array(all_sentences_test)

    def load(self):
        # Process raw:
        # the padding works the following way: the max sentence length per batch (in the colab tutorial)
        # Pytorch stores variables in a OHE way, using numericalization.
        raw_data, labels = self.load_raw_data()
        data_dict = self.process_raw(raw_data, labels)
        vocab = self.construct_vocab(data_dict)
        padded_train, padded_test = self.pad_and_prep(vocab, data_dict)

        return {"X_train" : padded_train, "y_train" : np.array(data_dict["y_train"]),
                "X_test" : padded_test, "y_test" : np.array(data_dict["y_test"])}, vocab


if __name__ == '__main__':
    a = Dataset_Loader()
    b = a.load()
    #print(b)




