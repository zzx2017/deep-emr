import os
import numpy
import pandas 
import glob
import itertools
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json

trainingset_files = glob.glob("./data/training/*.txt")
trainingset = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in trainingset_files]).values

X_train = numpy.array([[int(y) for y in x.split(',')] for x in trainingset[:, 0]])
Y_train = numpy.array([[int(y) for y in str(x).split(',')] if str(x) != '0' else [] for x in trainingset[:, 1]])

empty_indices = numpy.array([i for i, x in enumerate(Y_train) if x == []])

X_train = numpy.delete(X_train, empty_indices)
Y_train = numpy.delete(Y_train, empty_indices)

mlb = MultiLabelBinarizer()
encoded_Y = mlb.fit_transform(Y_train)

nb_words = 30551
max_length = 922

X_train = sequence.pad_sequences(X_train, maxlen=max_length)

model = Sequential()
model.add(Embedding(nb_words, 32, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(254, activation='relu'))
model.add(Dense(38, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, encoded_Y, validation_data=(X_train, encoded_Y), epochs=60, batch_size=32)

model_json = model.to_json()
with open("cnn-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("cnn-model.h5")
print("Saved model to disk")

def confusion_matrix(truth, prediction):
	matrices = list()
	results = numpy.array([0, 0, 0])
	for i in range(len(predictions)):
		confusion = {'tp': 0, 'fp': 0, 'fn': 0}
		for j in range(len(predictions[i])):
			if predictions[i][j] in truth[i]:
				confusion['tp'] = confusion['tp'] + 1
			else:
				confusion['fp'] = confusion['fp'] + 1
		for j in range(len(truth[i])):
			if truth[i][j] not in predictions[i]:
				confusion['fn'] = confusion['fn'] + 1
		matrices.append(numpy.array([confusion['tp'], confusion['fp'], confusion['fn']]))
	for matrix in matrices:
		results = numpy.add(results, matrix)
	return results

def evaluate(matrix):
	tp, fp, fn = matrix[0], matrix[1], matrix[2]
	recall = tp / (tp + fn)
	precision = tp / (tp + fp) 
	f1 = 2 * ((precision * recall) / (precision + recall))
	return {'recall': recall, 'precision': precision, 'f1': f1}

testset_files = glob.glob("./data/test/text/*.txt")
label_files = glob.glob("./data/test/labels/*.txt")
testset = [(pandas.read_csv(x, delim_whitespace=True, header=None)).values for x in testset_files]
labels = pandas.concat([(pandas.read_csv(x, delim_whitespace=True, header=None)) for x in label_files]).values

X_test = [[[int(z) for z in str(y[0]).split(',')] for y in x] for x in testset]
Y_test = [[int(y) for y in str(x[0]).split(',')] for x in labels]
Y_test = [x if x[0] != 0 else [] for x in Y_test]

X_test = [sequence.pad_sequences(x, maxlen=max_length) for x in X_test]

json_file = open('cnn-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnn-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = list()
for record in X_test:
	prediction = loaded_model.predict(record)
	prediction = numpy.array([[round(y) for y in x] for x in prediction])
	predicted_Y = mlb.inverse_transform(prediction)
	predicted_Y = list(set([y for x in predicted_Y for y in x]))
	predictions.append(predicted_Y)

matrix = confusion_matrix(Y_test, predictions)
performance = evaluate(matrix)

print(matrix)
print(performance)
