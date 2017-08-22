import numpy
import pandas 
import glob
import gensim
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.optimizers import Nadam
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
# from keras.utils import plot_model  
from sklearn.preprocessing import LabelEncoder
from collections import Counter

dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
classes = pandas.read_csv("../../data/classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(0)[1].to_dict()

files = glob.glob("./data/training/*.txt")
training_set = [pandas.read_csv(file, delim_whitespace=True, header=None).values for file in files]

train_records, train_labels = [], []
X_train, Y_train = [], []

for record in training_set:
	train_records.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 0]]))
	train_labels.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 1]]))

for i in range(len(train_records)):
	X_train.append([])
	Y_train.append([])
	for j in range(len(train_records[i])):
		X_train[i].extend(train_records[i][j])
		Y_train[i].extend(train_labels[i][j])
	X_train[i] = numpy.array(X_train[i])
	Y_train[i] = numpy.array(Y_train[i])

nb_words = 36664
max_length = 3390
embedding_dim = 20

y_train = [list(set(x)) for x in Y_train]
y_train = [y for x in y_train for y in x]
y_train = [x for x in y_train if x != 1]

X_train = sequence.pad_sequences(X_train, maxlen=max_length)
Y_train = sequence.pad_sequences(Y_train, maxlen=max_length)

Y_train = numpy.array([y for x in Y_train for y in x])
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
encoded_Y = np_utils.to_categorical(encoded_Y)
encoded_Y = numpy.array([encoded_Y[i:i + max_length] for i in range(0, len(encoded_Y), max_length)])

word2vec_model = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
embedding_weights = numpy.zeros((nb_words, embedding_dim))

for word, index in dictionary.items():
	if word in word2vec_model:
		embedding_weights[index,:] = word2vec_model[word]

model = Sequential()
model.add(Embedding(nb_words, embedding_dim, input_length=max_length, mask_zero=True, weights=[embedding_weights]))
model.add(GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(TimeDistributed(Dense(encoded_Y.shape[2], activation='softmax')))

optimiser = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=optimiser) 

# plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True) 

print(model.summary())
print(model.get_config())

# early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)
# model.fit(X_train, encoded_Y, epochs=75, batch_size=32, callbacks=[early_stopping_monitor], verbose=2)
model.fit(X_train, encoded_Y, epochs=40, batch_size=32, verbose=2)

model_json = model.to_json()
with open("gru-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("gru-model.h5")
print("Saved model to disk")

def confusion_matrix(truth, predictions, report):
	matrices = list()
	results = numpy.array([0, 0, 0])
	for i in range(len(predictions)):
		confusion = {'tp': 0, 'fp': 0, 'fn': 0}
		for j in range(len(predictions[i])):
			if predictions[i][j] in truth[i]:
				confusion['tp'] = confusion['tp'] + 3 if 'continuing' in classes[predictions[i][j]] else confusion['tp'] + 1
				report[predictions[i][j]][0] = report[predictions[i][j]][0] + 1
			else:
				confusion['fp'] = confusion['fp'] + 3 if 'continuing' in classes[predictions[i][j]] else confusion['fp'] + 1
				report[predictions[i][j]][1] = report[predictions[i][j]][1] + 1
		for j in range(len(truth[i])):
			if truth[i][j] not in predictions[i]:
				confusion['fn'] = confusion['fn'] + 3 if 'continuing' in classes[truth[i][j]] else confusion['fn'] + 1
				report[truth[i][j]][2] = report[truth[i][j]][2] + 1
		matrices.append(numpy.array([confusion['tp'], confusion['fp'], confusion['fn']]))
	for matrix in matrices:
		results = numpy.add(results, matrix)
	return results

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
		x_test[i].extend(X_test[i][j])
	x_test[i] = numpy.array(x_test[i])

x_test = sequence.pad_sequences(x_test, maxlen=max_length)

json_file = open('gru-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("gru-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer=optimiser)

predictions = loaded_model.predict(x_test)
predictions = numpy.array([[[round(z) for z in y] for y in x] for x in predictions])
predictions = [x.argmax(1) for x in predictions]
predictions = [list(set(x)) for x in predictions]
predictions = [[y for y in x if y != 0 and y != 1] for x in predictions]

train_label_count = Counter(y_train).most_common()
test_label_count = Counter([y for x in y_test for y in x]).most_common()
train_samples = {x[0]: x[1] for x in train_label_count}
expected = {x[0]: x[1] for x in test_label_count}
report = {x: [0, 0, 0] for x in range(2, 104)}

matrix = confusion_matrix(y_test, predictions, report)
smoker_fp = report[5][2] + report[14][2] + report[18][2] + report[46][2]
smoker_fn = report[5][1] + report[14][1] + report[18][1] + report[46][1]
smoker_tp = len(y_test) - expected[5] - expected[14] - expected[18] - expected[46] - smoker_fn
family_hist_fp = report[24][2]
family_hist_fn = report[24][1]
family_hist_tp = len(y_test) - expected[24] - family_hist_fn
matrix[0] = matrix[0] + smoker_tp + family_hist_tp
matrix[1] = matrix[1] + smoker_fp + family_hist_fp
matrix[2] = matrix[2] + smoker_fn + family_hist_fn

for k, v in report.items():
	tp, fp, fn = v[0], v[1], v[2]
	print("%s,%d,%d,%d,%d,%f,%f,%f,%d" % (classes[k][2::], expected[k] if k in expected else 0, tp, fp, fn, 0, 0, 0, train_samples[k] if k in train_samples else 0))

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