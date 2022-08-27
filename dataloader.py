import os
import pickle
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import sys
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from keras import callbacks
from keras import optimizers
import json
from helper import *
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *


class DataLoader:
    def __init__(self, config_passed):
        # self.config = config
        # if config["data_name"] is "flickr8k":
        #     self.flickr8k()
        # if config["data_name"] is "flickr8k_polish":
        #     self.flickr8k()
        # elif config["data_name"] is "flickr30k":
        #     self.coco_data()
        # elif config["data_name"] is "flickr30k_polish":
        #     self.flickr8k()
        # elif config["data_name"] is "aide":
        #     self.flickr8k()
        # elif config["data_name"] is "coco14":
        #     self.coco_data()
        # elif config["data_name"] is "coco17":
        #     self.coco_data()
        self.mixed(config_passed)

    def case_train(self, config_passed):
        if config_passed["train_images"] is "flickr8k":
            with open(config_flickr8k["encoded_images_train"], "rb") as encoded_pickle:
                encoding_train = load(encoded_pickle)
            descriptions = load_descriptions(config_flickr8k["token_path"])
            train_descriptions = load_clean_descriptions_new(config_flickr8k["preprocessed_descriptions_save_path"],
                                                                  list(encoding_train.keys()))
        elif config_passed["train_images"] is "flickr8k_polish":
            with open(config_flickr8k_polish["encoded_images_train"], "rb") as encoded_pickle:
                encoding_train = load(encoded_pickle)
            descriptions = load_descriptions(config_flickr8k_polish["token_path"])
            train_descriptions = load_clean_descriptions_new(config_flickr8k_polish["preprocessed_descriptions_save_path"],
                                                                  list(encoding_train.keys()))
        elif config_passed["train_images"] is "flickr30k":
            with open(config_flickr30k["encoded_images_train"], "rb") as encoded_pickle:
                encoding_train = load(encoded_pickle)
            descriptions = load_descriptions(config_flickr30k["token_path"])
            train_descriptions = load_clean_descriptions_new(config_flickr30k["preprocessed_descriptions_save_path"],
                                                                  list(encoding_train.keys()))
        elif config_passed["train_images"] is "flickr30k_polish":
            with open(config_flickr30k_polish["encoded_images_train"], "rb") as encoded_pickle:
                encoding_train = load(encoded_pickle)
            descriptions = load_descriptions(config_flickr30k_polish["token_path"])
            train_descriptions = load_clean_descriptions_new(config_flickr30k_polish["preprocessed_descriptions_save_path"],
                                                                  list(encoding_train.keys()))
        return encoding_train, descriptions, train_descriptions

    def case_test(self, config_passed):
        if config_passed["test_images"] is "flickr8k":
            with open(config_flickr8k["encoded_images_test"], "rb") as encoded_pickle:
                encoding_test = load(encoded_pickle)

        elif config_passed["test_images"] is "flickr8k_polish":
            with open(config_flickr8k_polish["encoded_images_test"], "rb") as encoded_pickle:
                encoding_test = load(encoded_pickle)

        elif config_passed["test_images"] is "flickr30k":
            with open(config_flickr30k["encoded_images_test"], "rb") as encoded_pickle:
                encoding_test = load(encoded_pickle)

        elif config_passed["test_images"] is "flickr30k_polish":
            with open(config_flickr30k_polish["encoded_images_test"], "rb") as encoded_pickle:
                encoding_test = load(encoded_pickle)

        return encoding_test

    def mixed(self, config_passed):
        self.image_features_train, self.descriptions, self.train_descriptions = self.case_train(config_passed)
        self.image_features_test = self.case_test(config_passed)
        self.out()

    def flickr8k(self):
        # load training dataset (6K)
        filename = self.config["train_images_path"]
        self.train = load_set(filename)
        print("Train dataset")
        #       id obrazow

        print('Train dataset loaded: %d' % len(self.train))
        # Below path contains all the images
        self.images = self.config["images_path"]
        # treningowy
        self.img = glob.glob(self.images + '*.jpg')
        self.train_img, self.train_images = self.images_with_path(self.config["train_images_path"])

        # zbior testowy
        self.test_img, self.test_images = self.images_with_path(self.config["test_images_path"])
        self.image_features_train, self.image_features_test = self.prepare_images(self.train_img, self.test_img)
        print('Photos: train=%d' % len(self.image_features_train))

        # załaduj opisy

        # parse descriptions
        self.descriptions = load_descriptions(self.config["token_path"])
        print('Loaded: %d ' % len(self.descriptions))

        clean_descriptions(self.descriptions, self.config)
        # summarize vocabulary
        print("Save descriptions in separate file")
        save_descriptions(self.descriptions, self.config["preprocessed_descriptions_save_path"])

        # descriptions
        self.train_descriptions = load_clean_descriptions_new(self.config["preprocessed_descriptions_save_path"],
                                                          list(self.image_features_train.keys()))
        print('Descriptions: train=%d' % len(self.train_descriptions))
#         self.out()

    def out(self):
        self.all_train_captions = self.all_train_captions(self.train_descriptions)
        print("Number of training captions ", len(self.all_train_captions))
        # determine the maximum sequence length
        self.max_length = max_length(self.train_descriptions)
        print('Description Length: %d' % self.max_length)
        # Count words and consider only words which occur at least 10 times in the corpus
        self.vocab = self.count_words_and_threshold(self.all_train_captions)
        self.ixtoword, self.wordtoix = self.ixtowordandbackward(self.vocab)
        self.vocab_size = len(self.ixtoword) + 1  # one for appended 0's
        print("Vocab size: ", self.vocab_size)

        self.embedding_matrix, self.embedding_vector = self.embedding_matrix(self.vocab_size, self.wordtoix)

    def all_train_captions(self, train_descriptions):
        # Create a list of all the training captions
        all_train_captions = []
        for key, val in train_descriptions.items():
            for cap in val:
                all_train_captions.append(cap)
        return all_train_captions

    def images_with_path(self, images_path):
        # Below file conatains the names of images to be used in test/train data
        # Read the validation image names in a set# Read the test/train image names in a set
        list_images = set(open(images_path, 'r').read().strip().split('\n'))
        # Create a list of all the test/train images with their full path names
        list_img = []
        for i in self.img:  # img is list of full path names of all images
            if i[len(self.images):] in list_images:  # Check if the image belongs to test/training set
                list_img.append(i)  # Add it to the list of test/train images
        return list_img, list_images

    def count_words_and_threshold(self, all_train_captions):
        word_count_threshold = 10
        word_counts = {}
        nsents = 0
        for sent in self.all_train_captions:
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
            ixtoword = load(encoded_pickle)
        with open(self.config["wordtoix_path"], "rb") as encoded_pickle:
            wordtoix = load(encoded_pickle)
        return ixtoword, wordtoix

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
                    print(index)
            if not os.path.isdir("./" + self.config["data_name"]):
                os.makedirs("./" + self.config["data_name"])
                os.makedirs("./" + self.config["data_name"] + "/Pickle")
            with open(self.config["encoded_images_test"], "wb") as encoded_pickle:
                pickle.dump(encoding_test, encoded_pickle)
            print(encoding_test)
            print("Test images encoded")
            print("Time taken in seconds =", time() - start)

            start = time()
            encoding_train = {}
            index = 1
            for img in train_img:
                image_filename = img.rsplit("/", 1)[-1]
                encoding_train[image_filename] = encode(img, images_feature_model)
                if index % 100 == 0:
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

    def embedding_matrix(self, vocab_size, wordtoix):
        # Load Glove vectors

        embeddings_index = {}  # empty dictionary

        f = open(self.config["word_embedings_path"], encoding="utf-8")
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
        embedding_matrix = np.zeros((vocab_size, self.config["embedings_dim"]))
        for word, i in wordtoix.items():
            # if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                # 1655,299 199
                embedding_matrix[i] = embedding_vector
        return embedding_matrix, embedding_vector

    def coco_data(self):
        self.train_img, self.train_images, _, _ = self.load_coco_data()

        _, _, self.test_img, self.test_images = self.load_coco_data()
        self.image_features_train, self.image_features_test = self.prepare_images(self.train_img, self.test_img)

        self.train_descriptions, self.descriptions = self.load_clean_descriptions_coco(self.config["token_path"],
                                                                                       self.train_images)
        print('Descriptions: train=%d' % len(self.train_descriptions))

        self.out()

    def desc_raw(self, imgs, train_images):
        descriptions_raw = dict()
        for img in imgs:
            image_filename = img['filename']
            # create list
            if image_filename.rsplit(".", 1)[0] not in descriptions_raw:
                descriptions_raw[image_filename.rsplit(".", 1)[0]] = list()
            for sent in img['sentences']:
                descriptions_raw[image_filename.rsplit(".", 1)[0]].append(sent['raw'])
        return descriptions_raw

    def load_clean_descriptions_coco(self, filename, train_images):
        imgs = json.load(open(filename, 'r'))
        imgs = imgs['images']
        descriptions = dict()

        if self.config["preprocess_descriptions"]:
            for img in imgs:
                image_filename = img['filename']
                if image_filename in train_images:
                    # create list
                    if image_filename.rsplit(".", 1)[0] not in descriptions:
                        descriptions[image_filename.rsplit(".", 1)[0]] = list()
                    for sent in img['sentences']:
                        # wrap descriion in tokens
                        desc = 'startseq ' + " ".join(sent['tokens']) + ' endseq'
                        # store
                        descriptions[image_filename.rsplit(".", 1)[0]].append(desc)
            with open(self.config["preprocessed_descriptions_save_path"], "wb") as encoded_pickle:
                pickle.dump(descriptions, encoded_pickle)

            return descriptions, self.desc_raw(imgs, train_images)

        with open(self.config["preprocessed_descriptions_save_path"], "rb") as encoded_pickle:
            descriptions = load(encoded_pickle)

        return descriptions, self.desc_raw(imgs, train_images)

    def load_coco_data(self):
        input_json = self.config["images_names_path"]
        images_folder = self.config["images_folder"]
        print('DataLoader loading json file: ', input_json)
        info = json.load(open(input_json))
        train_img = []
        train_images = []
        test_img = []
        test_images = []
        for ix in range(len(info['images'])):
            img = info['images'][ix]
            image_filename = img['file_path']
            file_path = images_folder + "/" + img['file_path']

            if image_filename.find("/") != -1:
                image_filename = img['file_path'].rsplit("/", 1)[1]

            if img['split'] == 'train':
                train_img.append(file_path)
                train_images.append(image_filename)
            elif img['split'] == 'val':
                test_img.append(file_path)
                test_images.append(image_filename)
            elif img['split'] == 'test':
                test_img.append(file_path)
                test_images.append(image_filename)
            elif img['split'] == 'restval':
                train_img.append(file_path)
                train_images.append(image_filename)

        return train_img, train_images, test_img, test_images
