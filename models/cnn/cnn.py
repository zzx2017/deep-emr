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

def ngram(s, n):
	s = [0]*int(n/2) + s + [0]*int(n/2)
	return [numpy.array(x) for x in zip(*[s[i:] for i in range(n)])]

dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
classes = pandas.read_csv("../../data/classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(0)[1].to_dict()

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

def confusion_matrix(truth, predictions):
	matrices = list()
	results = numpy.array([0, 0, 0])
	for i in range(len(predictions)):
		confusion = {'tp': 0, 'fp': 0, 'fn': 0}
		for j in range(len(predictions[i])):
			if predictions[i][j] in truth[i]:
				confusion['tp'] = confusion['tp'] + 3 if 'continuing' in classes[predictions[i][j]] else confusion['tp'] + 1
			else:
				confusion['fp'] = confusion['fp'] + 3 if 'continuing' in classes[predictions[i][j]] else confusion['fp'] + 1
		for j in range(len(truth[i])):
			if truth[i][j] not in predictions[i]:
				confusion['fn'] = confusion['fn'] + 3 if 'continuing' in classes[truth[i][j]] else confusion['fn'] + 1
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

test_files = glob.glob("./data/test/gold/*.txt")
test_set = [(pandas.read_csv(x, delim_whitespace=True, header=None)).values for x in test_files]

X_test = [[[int(z) for z in str(y[0]).split(',')] for y in x] for x in test_set]
Y_test = [x[-1] for x in X_test]
X_test = [x[0:-1] for x in X_test]
Y_test = [x if x[0] != 0 else [] for x in Y_test]

X_test = numpy.array(X_test)
Y_test = numpy.array(Y_test)

x_test, y_test = [], Y_test

for i in range(len(X_test)):
	x_test.append([])
	for j in range(len(X_test[i])):
		x_test[i].extend(ngram(X_test[i][j], n))
	x_test[i] = numpy.array(x_test[i])

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

matrix = confusion_matrix(y_test, predictions)
performance = evaluate(matrix)

print(matrix)
print(performance)

for i in range(len(test_files)):
	prediction = [classes[x][2::] for x in predictions[i]]

	file = open("output/" + test_files[i][17:24] + "xml", 'w')
	file.write("<?xml version='1.0' encoding='UTF-8'?>\n")
	file.write("<root>\n")
	file.write("\t<TAGS>\n")

	for label in prediction:
		label = label.split('.')
		if len(label) == 3:
			if label[2] == 'continuing':
				if label[0] == 'medication':
					element = label[0].upper()
					type1 = label[1].replace('_', ' ')
					file.write('\t\t<' + element + ' time="before dct" type1="' + type1 + '" type2=""/>\n')
					file.write('\t\t<' + element + ' time="during dct" type1="' + type1 + '" type2=""/>\n')
					file.write('\t\t<' + element + ' time="after dct" type1="' + type1 + '" type2=""/>\n')
				else:
					element = label[0].upper()
					indicator = label[1].replace('_', ' ')
					file.write('\t\t<' + element + ' time="before dct" indicator="' + indicator + '"/>\n')
					file.write('\t\t<' + element + ' time="during dct" indicator="' + indicator + '"/>\n')
					file.write('\t\t<' + element + ' time="after dct" indicator="' + indicator + '"/>\n')
			else:
				if label[0] == 'medication':
					element = label[0].upper()
					time = label[2].replace('_', ' ')
					type1 = label[1].replace('_', ' ')
					file.write('\t\t<' + element + ' time="' + time + '" type1="' + type1 + '" type2=""/>\n')
				else:
					element = label[0].upper()
					time = label[2].replace('_', ' ')
					indicator = label[1].replace('_', ' ')
					file.write('\t\t<' + element + ' time="' + time + '" indicator="' + indicator + '"/>\n')
		elif len(label) == 2:
			if label[0] == 'smoker':
				element = label[0].upper()
				status = label[1]
				file.write('\t\t<' + element + ' status="' + status + '"/>\n')
			elif label[0] == 'family_hist':
				element = label[0].upper()
				indicator = label[1]
				file.write('\t\t<' + element + ' indicator="' + indicator + '"/>\n')

	if 'smoker.current' not in prediction and 'smoker.ever' not in prediction and 'smoker.never' not in prediction and 'smoker.past' not in prediction:
		file.write('\t\t<SMOKER status="unknown"/>\n')
	if 'family_hist.present' not in prediction:
		file.write('\t\t<FAMILY_HIST indicator="not present"/>\n')

	file.write("\t</TAGS>\n")
	file.write("</root>\n")
	file.close()
