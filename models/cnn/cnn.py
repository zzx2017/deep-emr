import math
import numpy
import pandas 
import glob
import gensim
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.optimizers import Nadam
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from collections import Counter

def ngram(s, n):
	s = [0]*int(n/2) + s + [0]*int(n/2)
	return [numpy.array(x) for x in zip(*[s[i:] for i in range(n)])]

dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
training_files = glob.glob("./data/training/*.txt")
training = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in training_files]).values

X_train = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 0]])
Y_train = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 1]])

empty_indices = numpy.array([i for i, x in enumerate(Y_train) if len(list(set(x))) == 1 and list(set(x))[0] == 1])

X_train = numpy.delete(X_train, empty_indices)
Y_train = numpy.delete(Y_train, empty_indices)

x_train, y_train = [], []
n = 5

for i in range(len(X_train)):
	x_train.extend(ngram(X_train[i], n))
	for y in ngram(Y_train[i], n):
		y_train.append([y[int(n/2)]])

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)

mlb = MultiLabelBinarizer()
encoded_y = mlb.fit_transform(y_train)

nb_words = 36664
max_length = 5
embedding_dim = 20

word2vec_model = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
embedding_weights = numpy.zeros((nb_words, embedding_dim))

for word, index in dictionary.items():
	if word in word2vec_model:
		embedding_weights[index, :] = word2vec_model[word]

dropout_prob = [0.2, 0.2]
filter_sizes = [2, 3, 4]

input_shape = (max_length,)
model_input = Input(shape=input_shape)

embedding_layer = Embedding(nb_words, embedding_dim, input_length=max_length, name="embedding", weights=[embedding_weights])(model_input)
embedding_layer = Dropout(dropout_prob[0])(embedding_layer)

convs = []
for filter_size in filter_sizes:
    conv = Convolution1D(filters=32, kernel_size=filter_size, padding="valid", activation="relu", strides=1)(embedding_layer)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    convs.append(conv)

merge = Concatenate()(convs)
merge = Dropout(dropout_prob[1])(merge)
dense = Dense(256, activation="relu")(merge)

model_output = Dense(encoded_y.shape[1], activation="softmax")(dense)
model = Model(model_input, model_output)

optimiser = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

print(model.summary())
print(model.get_config())

early_stopping_monitor = EarlyStopping(monitor='loss', patience=2)
model.fit(x_train, encoded_y, epochs=10, batch_size=32, callbacks=[early_stopping_monitor], verbose=2)

model_json = model.to_json()
with open("cnn-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("cnn-model.h5")
print("Saved model to disk")

def confusion_matrix(truth, predictions, report):
	matrices = list()
	results = numpy.array([0, 0, 0])
	for i in range(len(predictions)):
		confusion = {'tp': 0, 'fp': 0, 'fn': 0}
		for j in range(len(predictions[i])):
			if predictions[i][j] in truth[i]:
				confusion['tp'] = confusion['tp'] + 1
				report[predictions[i][j]][0] = report[predictions[i][j]][0] + 1
			else:
				confusion['fp'] = confusion['fp'] + 1
				report[predictions[i][j]][1] = report[predictions[i][j]][1] + 1
		for j in range(len(truth[i])):
			if truth[i][j] not in predictions[i]:
				confusion['fn'] = confusion['fn'] + 1
				report[truth[i][j]][2] = report[truth[i][j]][2] + 1
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

test_files = glob.glob("./data/test/*.txt")
test_set = [(pandas.read_csv(x, delim_whitespace=True, header=None)).values for x in test_files]

X_test, Y_test = [], []

for record in test_set:
	X_test.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 0]]))
	Y_test.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 1]]))

X_test = numpy.array(X_test)
Y_test = numpy.array(Y_test)

x_test, y_test = [], []

for i in range(len(X_test)):
	x_test.append([])
	for j in range(len(X_test[i])):
		x_test[i].extend(ngram(X_test[i][j], n))
	x_test[i] = numpy.array(x_test[i])

for record in Y_test:
	labels = [y for x in record for y in x]
	y_test.append(list(set([x for x in labels if x != 1])))

exception = [83, 84, 94, 96, 102, 103]
y_test = [[y for y in x if y not in exception] for x in y_test]

json_file = open('cnn-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnn-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

predictions = list()
for record in x_test:
	prediction = loaded_model.predict(record)
	prediction = numpy.array([[round(y) for y in x] for x in prediction])
	prediction = mlb.inverse_transform(prediction)
	prediction = list(set([y for x in prediction for y in x]))
	prediction = [x for x in prediction if x != 1]
	predictions.append(prediction)

classes = pandas.read_csv("../../data/classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(0)[1].to_dict()

label_count = Counter([y for x in y_test for y in x]).most_common()
expected = {x[0]: x[1] for x in label_count}
report = {x: [0, 0, 0] for x in range(2, 104)}

matrix = confusion_matrix(y_test, predictions, report)
performance = evaluate(matrix)

file = open("cnn-performance.csv", 'w')
for k, v in report.items():
	tp, fp, fn = v[0], v[1], v[2]
	file.write("%s,%d,%d,%d,%d,%f,%f,%f\n" % (classes[k][2::], expected[k] if k in expected else 0, tp, fp, fn, 0, 0, 0))
file.close()

print(performance)
