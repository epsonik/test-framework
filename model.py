
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
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
from helper import *
from pickle import dump, load
from eval_utils import calculate_results
class ModelImpl:
    def __init__(self, data):
        self.data=data
        self.config = data.config
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.data.max_length,))
        se1 = Embedding(self.data.vocab_size, self.config["embedings_dim"], mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.data.vocab_size, activation='softmax')(decoder2)
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.summary()
        self.model.layers[2]
        
        self.model.layers[2].set_weights([data.embedding_matrix])
        self.model.layers[2].trainable = False
        
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer())
        self.setup()
        
        
    def optimizer(self):
        return Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    def setup(self):
            # model.optimizer.lr = 0.001
        self.epochs = 2
        self.number_pics_per_bath = 100
        self.steps = len(self.data.train_descriptions)//self.number_pics_per_bath
        
    def train(self):
        if self.config["train_model"]:
            callback = callbacks.EarlyStopping(monitor='loss',min_delta = 0.001, patience=3)
            generator = data_generator(self.data.train_descriptions, 
                                       self.data.image_features_train, 
                                       self.data.wordtoix, 
                                       self.data.max_length, 
                                       self.number_pics_per_bath, 
                                       self.data.vocab_size)
            self.model.fit(generator, epochs=self.epochs,
                           steps_per_epoch=self.steps,
                           callbacks=[callback],
                           verbose=1)
            if self.config["save_model"]:
                writepath = self.config["lstm_model_save_dir"] + 'model' + '.h5'
                self.model.save(writepath)
                self.model.save_weights(self.config["lstm_model_save_path"])
        else:
            self.model.load_weights(self.config["lstm_model_save_path"])
    
    def evaluate(self):
        with open(self.config["encoded_images_test"], "rb") as encoded_pickle:
            encoding_test = load(encoded_pickle)
        expected, results = prepare_for_evaluation(encoding_test, self.data, self.model)
        out = calculate_results(expected, results, self.config)
        print(out)