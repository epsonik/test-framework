import pickle
from time import time
import numpy as np
from config import general
import os
import string

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
import itertools


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_embedding_matrix(vocab_size, wordtoix, word_embedings_path, embedings_dim):
    # Load Glove vectors

    embeddings_index = {}  # empty dictionary

    f = open(word_embedings_path, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        import re
        if isfloat(values[1]):
            coefs = np.asarray(values[2:], dtype='float32')
        elif isfloat(values[2]):
            coefs = np.asarray(values[3:], dtype='float32')
        elif isfloat(values[3]):
            coefs = np.asarray(values[4:], dtype='float32')
        elif isfloat(values[4]):
            coefs = np.asarray(values[5:], dtype='float32')
        else:
            coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedings_dim))
    for word, i in wordtoix.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            # 1655,299 199
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, embedding_vector


def define_images_feature_model():
    # Load the inception v3 model
    model = InceptionV3(weights='imagenet')
    # Create a new model, by removing the last layer (output layer) from the inception v3
    model_new = Model(model.input, model.layers[-2].output)
    return model_new


def prepare_images(self, train_img, test_img):
    # Call the funtion to encode all the train images
    # This will take a while on CPU - Execute this only once
    images_feature_model = define_images_feature_model()
    if self.config["encode_images"]:
        start = time()
        encoding_test = {}
        index = 1
        for img in test_img:
            image_filename = img.rsplit("/", 1)[-1]
            encoding_test[image_filename] = encode(img, images_feature_model)
            if index % 100 == 0:
                print("Processed:")
                print(index)
        if not os.path.isdir("./" + self.config["data_name"]):
            os.makedirs("./" + self.config["data_name"])
            os.makedirs("./" + self.config["data_name"] + "/Pickle")
        with open(self.config["encoded_images_test"], "wb") as encoded_pickle:
            pickle.dump(encoding_test, encoded_pickle)
        print("Test images encoded")
        print("Time taken in seconds =", time() - start)

        start = time()
        encoding_train = {}
        index = 1
        for img in train_img:
            image_filename = img.rsplit("/", 1)[-1]
            encoding_train[image_filename] = encode(img, images_feature_model)
            if index % 100 == 0:
                print("Processed:")
                print(index)
            index += 1
        mode = 'a' if os.path.exists(self.config["encoded_images_train"]) else 'wb'
        # Save the bottleneck train features to disk
        with open(self.config["encoded_images_train"], 'w+b') as encoded_pickle:
            pickle.dump(encoding_train, encoded_pickle)
        print("Train images encoded")
        print("Time taken in seconds =", time() - start)

        return encoding_train, encoding_test

    with open(self.config["encoded_images_train"], "rb") as encoded_pickle:
        encoding_train = load(encoded_pickle)
    with open(self.config["encoded_images_test"], "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)
    return encoding_train, encoding_test


def clean_descriptions(captions_mapping):
    """Method to:
    *lower all letters
    *remove punctuation
    *remove tokens with numbers
    Parameters
    ----------
    captions_mapping : dict
        Dictionary with keys - image_id and values-list of ground truth captions from training set.

    Returns
    -------
    """
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in captions_mapping.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            # desc_list[i] = ' '.join(desc)
            desc_list[i] = desc


def wrap_captions_in_start_stop(training_captions):
    """Method to wrap captions into START and STOP tokens
    Parameters
    ----------
    training_captions : dict
        Dictionary with keys - image_id and values-list od ground truth captions

    Returns
    -------
    train_captions_preprocessed: dict-
            Dictionary with wrapped into START and STOP tokens captions.
    """
    train_captions_preprocessed = dict()
    for image_id in training_captions.keys():
        sentences = training_captions[image_id]
        if image_id not in train_captions_preprocessed:
            train_captions_preprocessed[image_id] = list()
        for sentence in sentences:
            # wrap descriion in START and STOP tokens
            desc = general['START'] + " ".join(sentence) + general['STOP']
            # store
            train_captions_preprocessed[image_id].append(desc)
    return train_captions_preprocessed


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to encode a given image into a vector of size (2048, )
def encode(image_path, images_feature_model):
    image = preprocess(image_path)  # preprocess the image
    fea_vec = images_feature_model.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec


def preprocess_images(train_images, test_images, configuration):
    # Call the funtion to encode all the train images
    # This will take a while on CPU - Execute this only once
    images_feature_model = define_images_feature_model()
    if configuration["encode_images"]:
        start = time()
        encoding_test = {}
        index = 1
        for image_id, image_path in test_images.items():
            encoding_test[image_id] = encode(image_path, images_feature_model)
            if index % 100 == 0:
                print("Processed:")
                print(index)
        print("Test images encoded")
        print("Time taken in seconds =", time() - start)

        mode = 'a' if os.path.exists(configuration["encoded_images_test_path"]) else 'wb'
        # Save the bottleneck train features to disk
        with open(configuration["encoded_images_test_path"], 'w+b') as encoded_pickle:
            pickle.dump(encoding_test, encoded_pickle)

        print("Train images encoded")
        print("Time taken in seconds =", time() - start)
        start = time()
        encoding_train = {}
        index = 1
        for image_id, image_path in train_images.items():
            encoding_train[image_id] = encode(image_path, images_feature_model)
            if index % 100 == 0:
                print("Processed:")
                print(index)
            index += 1
        mode = 'a' if os.path.exists(configuration["encoded_images_train_path"]) else 'wb'
        # Save the bottleneck train features to disk
        with open(configuration["encoded_images_train_path"], 'w+b') as encoded_pickle:
            pickle.dump(encoding_train, encoded_pickle)
        print("Train images encoded")
        print("Time taken in seconds =", time() - start)

        return encoding_train, encoding_test

    with open(configuration["encoded_images_train_path"], "rb") as encoded_pickle:
        encoded_images_train = pickle.load(encoded_pickle)
    with open(configuration["encoded_images_test_path"], "rb") as encoded_pickle:
        encoded_images_test = pickle.load(encoded_pickle)
    return encoded_images_train, encoded_images_test


def get_all_train_captions_list(train_captions):
    # Create a list of all the training captions
    all_train_captions = list(itertools.chain(*train_captions.values()))
    return all_train_captions


# calculate the length of the description with the most words
def get_max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def count_words_and_threshold(all_train_captions):
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    # Consider only words which occur at least 10 times in the corpus
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    return vocab


def ixtowordandbackward(self, vocab):
    if self.config["save_ix_to_word"]:
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        with open(self.config["ixtoword_path"], "wb") as encoded_pickle:
            pickle.dump(ixtoword, encoded_pickle)
        with open(self.config["wordtoix_path"], "wb") as encoded_pickle:
            pickle.dump(wordtoix, encoded_pickle)
        return ixtoword, wordtoix

    with open(self.config["ixtoword_path"], "rb") as encoded_pickle:
        ixtoword = pickle.load(encoded_pickle)
    with open(self.config["wordtoix_path"], "rb") as encoded_pickle:
        wordtoix = pickle.load(encoded_pickle)
    return ixtoword, wordtoix


def define_learning_data(data):
    def get_split(split, subset_data):
        if data.configuration[split]['subset_name'] == 'train':
            return subset_data["train"]['train_images_mapping_original'], \
                   subset_data["train"]['train_captions_mapping_original']
        if data.configuration[split]['subset_name'] == 'test':
            return subset_data["test"]['test_images_mapping_original'], \
                   subset_data["test"]['test_captions_mapping_original']

    train_images_mapping, train_captions_mapping = get_split("train", data.train)
    test_images_mapping, test_captions_mapping = get_split("test", data.test)

    return train_images_mapping, train_captions_mapping, test_images_mapping, test_captions_mapping, data.train[
        "all_captions"]


def preprocess_data(data):
    train_images_mapping, \
    train_captions_mapping, \
    test_images_mapping, \
    test_captions_mapping, \
    all_captions = define_learning_data(data)
    clean_descriptions(train_captions_mapping)
    train_captions_preprocessed = wrap_captions_in_start_stop(train_captions_mapping)
    print(train_captions_preprocessed)
    preprocess_images(train_images, test_images, configuration)
    all_train_captions = get_all_train_captions_list(train_captions_preprocessed)
    print("Number of training captions ", len(all_train_captions))
    # determine the maximum sequence length
    max_length = get_max_length(all_train_captions)
    print('Description Length: %d' % max_length)
    # Count words and consider only words which occur at least 10 times in the corpus
    vocab = count_words_and_threshold(all_train_captions)
    ixtoword, wordtoix = ixtowordandbackward(vocab)
    vocab_size = len(ixtoword) + 1  # one for appended 0's
    print("Vocab size: ", vocab_size)

    embedding_matrix, embedding_vector = get_embedding_matrix(vocab_size, wordtoix,
                                                              configuration["word_embedings_path"],
                                                              configuration["embedings_dim"])
