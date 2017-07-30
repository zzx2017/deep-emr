from __future__ import print_function
import math
import numpy
import pandas 
import glob
import gensim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.optimizers import Nadam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():
	training_files = glob.glob("./data/training/*.txt")
	training = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in training_files]).values

	X = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 0]])
	Y = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 1]])

	empty_indices = numpy.array([i for i, x in enumerate(Y) if len(list(set(x))) == 1 and list(set(x))[0] == 1])

	X = numpy.delete(X, empty_indices)
	Y = numpy.delete(Y, empty_indices)

	x_train, y_train = [], []
	n = 5

	for i in range(len(X)):
		X[i] = [0]*int(n/2) + X[i] + [0]*int(n/2)
		x_ngram = [numpy.array(x) for x in zip(*[X[i][j:] for j in range(n)])]
		x_train.extend(x_ngram)
		Y[i] = [0]*int(n/2) + Y[i] + [0]*int(n/2)
		y_ngram = [numpy.array(x) for x in zip(*[Y[i][j:] for j in range(n)])]
		for y in y_ngram:
			y_train.append([y[int(n/2)]])

	x_train = numpy.array(x_train)
	y_train = numpy.array(y_train)

	mlb = MultiLabelBinarizer()
	encoded_y = mlb.fit_transform(y_train)

	x_train, x_test, y_train, y_test = train_test_split(x_train, encoded_y, test_size=0.2)
	return x_train, y_train, x_test, y_test

def model(x_train, y_train, x_test, y_test):
	nb_words = 36664
	max_length = 5
	embedding_dim = 20

	dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
	dictionary = dictionary.set_index(1)[0].to_dict()

	word2vec_model = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
	embedding_weights = numpy.zeros((nb_words, embedding_dim))

	for word, index in dictionary.items():
		if word in word2vec_model:
			embedding_weights[index, :] = word2vec_model[word]

	filter_sizes = [2, 3, 4]

	input_shape = (max_length,)
	model_input = Input(shape=input_shape)

	embedding_layer = Embedding(nb_words, embedding_dim, input_length=max_length, name="embedding", weights=[embedding_weights])(model_input)
	embedding_layer = Dropout({{choice([0.1, 0.2, 0.3, 0.4, 0.5])}})(embedding_layer)

	convs = []
	for filter_size in filter_sizes:
		conv = Convolution1D(filters={{choice([32, 64, 128])}}, kernel_size=filter_size, padding="valid", activation="relu", strides=1)(embedding_layer)
		conv = MaxPooling1D(pool_size=2)(conv)
		conv = Flatten()(conv)
		convs.append(conv)

	merge = Concatenate()(convs)
	merge = Dropout({{choice([0.1, 0.2, 0.3, 0.4, 0.5])}})(merge)
	dense = Dense({{choice([128, 256, 512])}}, activation="relu")(merge)

	model_output = Dense(encoded_y.shape[1], activation="softmax")(dense)
	model = Model(model_input, model_output)

	optimiser = Nadam(lr={{choice([0.001, 0.002, 0.003, 0.004])}}, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
	model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

	early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2)
	model.fit(x_train, y_train, epochs={{choice([10, 20, 30, 40, 50, 60, 70, 80 ,90, 100])}}, batch_size={{choice([32, 64, 128, 256])}}, verbose=2, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor])
	score, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score)
	print('Test accuracy:', acc)
	return {'loss': score, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
	best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)
