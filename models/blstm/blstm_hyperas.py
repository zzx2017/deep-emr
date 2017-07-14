from __future__ import print_function
import numpy
import pandas 
import glob
import gensim
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.optimizers import Nadam
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():
	files = glob.glob("./data/training/*.txt")
	training_set = [pandas.read_csv(file, delim_whitespace=True, header=None).values for file in files]

	train_records, train_labels = [], []
	X, Y = [], []

	for record in training_set:
		train_records.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 0]]))
		train_labels.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 1]]))

	for i in range(len(train_records)):
		X.append([])
		Y.append([])
		for j in range(len(train_records[i])):
			X[i].extend(train_records[i][j])
			Y[i].extend(train_labels[i][j])
		X[i] = numpy.array(X[i])
		Y[i] = numpy.array(Y[i])

	max_length = 3390

	X = sequence.pad_sequences(X, maxlen=max_length)
	Y = sequence.pad_sequences(Y, maxlen=max_length)

	Y = numpy.array([y for x in Y for y in x])
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	encoded_Y = np_utils.to_categorical(encoded_Y)
	encoded_Y = numpy.array([encoded_Y[i:i + max_length] for i in range(0, len(encoded_Y), max_length)])

	x_train, x_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2)
	return x_train, y_train, x_test, y_test

def model(x_train, y_train, x_test, y_test):
	nb_words = 36664
	max_length = 3390
	embedding_dim = 20

	dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
	dictionary = dictionary.set_index(1)[0].to_dict()

	word2vec_model = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
	embedding_weights = numpy.zeros((nb_words, embedding_dim))

	for word, index in dictionary.items():
		if word in word2vec_model:
			embedding_weights[index,:] = word2vec_model[word]

	model = Sequential()
	model.add(Embedding(nb_words, embedding_dim, input_length=max_length, mask_zero=True, weights=[embedding_weights]))
	model.add(Bidirectional(LSTM(256, activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softmax'])}}, recurrent_activation={{choice(['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softmax'])}}, return_sequences=True)))
	model.add(Dropout(0.1))
	model.add(TimeDistributed(Dense(encoded_Y.shape[2], activation='softmax')))

	optimiser = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
	model.compile(loss='categorical_crossentropy', optimizer=optimiser) 

	early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2)
	model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor])
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score)
	return {'loss': score, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
	best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=25, trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)
