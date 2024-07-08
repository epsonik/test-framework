#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataloader import *
from config import *
from data_processor import preprocess_data
from keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.models import Model
from keras import Input
from eval_utils import calculate_results, prepare_for_evaluation
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
import os
from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# In[ ]:

data = DataLoader(config_mixed_coco14_coco14_Xception_glove_concatenate_dense512)

# In[ ]:


data = preprocess_data(data)


# In[ ]:


def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch, vocab_size):
    """
    Data generator, that serves the data to the model during training


    Parameters
    ----------
    descriptions: str
        Dictionary where key is image id and value is a list of wraped in start and stop captions, lemmatized, without punctuation descriptions
    photos
        Dictionary with encoded images(vector of image features extracted by specified image feature extractor
         fe. Inception), identified by image id
    wordtoix
        Dictionary with keys-words , values -id of word
    max_length
        Max number of words in caption on dataset
    num_photos_per_batch: int
    vocab_size: int
    Returns
    -------
    """
    X1, X2, y = list(), list(), list()
    n = 0
    # iterujemy po opisach doobrazu
    # kazdy opis zamieniamy na wektor liczb za pomoca slownika wordtoix
    # tworzymy mase par (zdjęcie + slowa) i sekwencja wyjsciowa. Czyli na bazie czeci zdania i zdjecia przewidujemy reszte zdania
    while 1:
        for image_id, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature from the dictionary
            photo = photos[image_id]
            for desc in desc_list:
                # encode the sentence by translating it to the number representation,
                # with the dictionary of words created in in the previous stage
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n = 0


class ModelImpl:
    def __init__(self, data):
        self.data = data
        if data.configuration['images_processor'] == 'vgg16' or data.configuration['images_processor'] == 'vgg19':
            img_features_size = 4096
            inputs1 = Input(shape=(4096,))
        elif data.configuration['images_processor'] == 'denseNet121':
            img_features_size = 1024
            inputs1 = Input(shape=(1024,))
        elif data.configuration['images_processor'] == 'mobileNet':
            img_features_size = 1000
            inputs1 = Input(shape=(1000,))
        elif data.configuration['images_processor'] == 'mobileNetV2':
            img_features_size = 1280
            inputs1 = Input(shape=(1280,))
        elif data.configuration['images_processor'] == 'denseNet201':
            img_features_size = 1920
            inputs1 = Input(shape=(1920,))
        else:
            img_features_size = 2048
            inputs1 = Input(shape=(img_features_size,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.data.max_length,))
        # The Embedding layer can be understood as a lookup table that maps from integer
        # indices (which stand for specific words) to dense vectors (their embeddings).
        if data.configuration["text_processor"] == "fastText":
            se1 = Embedding(self.data.vocab_size, fastText[self.data.language]["embedings_dim"], mask_zero=True)(
                inputs2)
        else:
            se1 = Embedding(self.data.vocab_size, glove[self.data.language]["embedings_dim"], mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = concatenate([fe2, se3])
        decoder2 = Dense(512, activation='relu')(decoder1)
        outputs = Dense(self.data.vocab_size, activation='softmax')(decoder2)
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.summary()
        self.model.layers[2]

        self.model.layers[2].set_weights([self.data.embedding_matrix])
        self.model.layers[2].trainable = False

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer())
        self.setup()

    def optimizer(self):
        return Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    def setup(self):
        # model.optimizer.lr = 0.001
        self.epochs = 100
        self.number_pics_per_bath = 450
        self.steps = len(self.data.train_captions_wrapped) // self.number_pics_per_bath

    def load_weights(self, model_name):
        self.model.load_weights(model_name)

    def evaluate(self, model_name):
        self.expected, self.results = prepare_for_evaluation(self.data.encoded_images_test,
                                                             self.data.test_captions_mapping,
                                                             self.data.wordtoix, self.data.ixtoword,
                                                             self.data.max_length,
                                                             self.model, self.data.configuration["images_processor"])
        out = calculate_results(self.expected, self.results, self.data.configuration, model_name)


# In[ ]:

print(device_lib.list_local_devices())
weights_dir = "./" + data.configuration["data_name"] + "/" + data.configuration["model_save_dir"] + "/t1"
for model_name in os.listdir(weights_dir):
    if model_name.endswith(".h5") and (not model_name.startswith('.')):
        print(model_name)
        model = ModelImpl(data)
        model.load_weights(weights_dir + "/" + model_name)
        model.evaluate(model_name)
