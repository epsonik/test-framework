from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.merge import add
from keras.models import Model
from keras import Input, layers
from keras import callbacks
from helper import *
from eval_utils import calculate_results


class ModelImpl:
    def __init__(self, data):
        self.data = data
        self.config = self.data.config
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.data.max_length,))
        se1 = Embedding(self.data.vocab_size, self.data.embedings_dim, mask_zero=True)(inputs2)
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
        self.steps = len(self.data.train_descriptions) // self.number_pics_per_bath

    def train(self):
        if self.config["train_model"]:
            callback = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=3)
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
                writepath = "./" + self.config["data_name"] + "/" + 'model' + '.h5'
                self.model.save(writepath)
                self.model.save_weights("./" + self.config["data_name"] + "/" + self.config["lstm_model_save_path"])
        else:
            self.model.load_weights(self.config["lstm_model_save_path"])

    def evaluate(self):
        encoding_test = self.data.image_features_test
        expected, results = prepare_for_evaluation(encoding_test, self.data, self.model)
        out = calculate_results(expected, results, self.config)
        print(out)
