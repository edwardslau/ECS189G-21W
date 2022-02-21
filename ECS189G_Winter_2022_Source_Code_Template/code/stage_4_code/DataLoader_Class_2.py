from code.base_class.dataset import dataset
import numpy as np
import os, re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.utils import shuffle

class Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_raw_data(self):

        parent_path_backup = "/Users/jacquelinemitchell/Documents/ECS189G/sample_code/ECS189G" \
                      "-21W/ECS189G_Winter_2022_Source_Code_Template/" \
                      "data/stage_4_data/text_classification/"

        if self.dataset_source_folder_path is not None:
            parent_path = self.dataset_source_folder_path
        else:
            parent_path = parent_path_backup
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


        data["train"], labels["train"] = shuffle(data["train"], labels["train"])
        data["test"], labels["test"] = shuffle(data["test"], labels["test"])

        return data, labels

    def preprocess_helper(self, s):
        s = re.sub(r"[^\w\s]", '', s)
        s = re.sub(r"\s+", '', s)
        s = re.sub(r"\d", '', s)
        #s = re.sub(r"[^a-zA-Z0-9]", " ", s)

        return s

    def preprocess_strings(self, data_dict, labels_dict):
        all_words = []
        stop_words = set(stopwords.words("english"))

        a = 25000

        for sentence in data_dict["train"][:a]:
            for word in sentence.lower().split():
                word = self.preprocess_helper(word)
                if word not in stop_words and word != '':
                    all_words.append(word)

        vocab = Counter(all_words).most_common(1000)
        vocab = [item[0] for item in vocab]
        encoder_dict = {word:i+1 for i, word in enumerate(vocab)}

        X_train, X_test = [], []
        for sentence in data_dict["train"][:a]:
            encoded_sequence = [encoder_dict[self.preprocess_helper(word)] for word in sentence.lower().split()
                                if self.preprocess_helper(word) in encoder_dict.keys()]
            X_train.append(encoded_sequence)

        for sentence in data_dict["test"][:a]:
            encoded_sequence = [encoder_dict[self.preprocess_helper(word)] for word in sentence.lower().split()
                                if self.preprocess_helper(word) in encoder_dict.keys()]
            X_test.append(encoded_sequence)

        return {"X_train" : np.array(X_train),
                "X_test" : np.array(X_test),
                "y_train" : np.array(labels_dict["train"])[:a],
                "y_test" : np.array(labels_dict["test"])[:a]}, encoder_dict

    def prep_pad(self, data, max_sentence_length=200):
        padded_inputs = np.zeros((len(data), max_sentence_length), dtype=int)
        for i, sentence in enumerate(data):
            if len(sentence) != 0:
                padded_inputs[i, -len(sentence):] = sentence[:max_sentence_length]
        return padded_inputs

    def load(self):
        raw_data, raw_labels = self.load_raw_data()
        cleaned_data, vocab = self.preprocess_strings(raw_data, raw_labels)
        X_train = self.prep_pad(cleaned_data["X_train"])
        X_test = self.prep_pad(cleaned_data["X_test"])

        return {"X_train" : np.array(X_train),
                "X_test" : np.array(X_test),
                "y_train" : np.array(cleaned_data["y_train"]),
                "y_test" : np.array(cleaned_data["y_test"])}, vocab

if __name__ == "__main__":
    a = Dataset_Loader()
    # res, res2 = a.load_raw_data()
    # res3 = a.preprocess_strings(res, res2)
    # pad_train, pad_test = a.prep_pad(res3["X_train"]), a.prep_pad(res3["X_test"])
    # print(pad_train)
    res = a.load()
    print(res)