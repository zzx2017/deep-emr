import numpy
import pandas 
import glob
import gensim
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.optimizers import Nadam
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from collections import Counter

dictionary = pandas.read_csv("../../data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
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
model.add(SimpleRNN(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))
model.add(TimeDistributed(Dense(encoded_Y.shape[2], activation='softmax')))

optimiser = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=optimiser) 

print(model.summary())
print(model.get_config())

# early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)
# model.fit(X_train, encoded_Y, epochs=75, batch_size=32, callbacks=[early_stopping_monitor], verbose=2)
model.fit(X_train, encoded_Y, epochs=40, batch_size=32, verbose=2)

model_json = model.to_json()
with open("rnn-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("rnn-model.h5")
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

test_records, test_labels = [], []
X_test, Y_test = [], []

for record in test_set:
	test_records.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 0]]))
	test_labels.append(numpy.array([[int(y) for y in x.split(',')] for x in record[:, 1]]))

for i in range(len(test_records)):
	X_test.append([])
	Y_test.append([])
	for j in range(len(test_records[i])):
		X_test[i].extend(test_records[i][j])
		Y_test[i].extend(test_labels[i][j])
	X_test[i] = numpy.array(X_test[i])
	Y_test[i] = numpy.array(Y_test[i])

y_test = []
for record in Y_test:
	y_test.append(list(set([x for x in record if x != 1])))

exception = [83, 84, 94, 96, 102, 103]
y_test = [[y for y in x if y not in exception] for x in y_test]

X_test = sequence.pad_sequences(X_test, maxlen=max_length)

json_file = open('rnn-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("rnn-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer=optimiser)

predictions = loaded_model.predict(X_test)
predictions = numpy.array([[[round(z) for z in y] for y in x] for x in predictions])
predictions = [x.argmax(1) for x in predictions]
predictions = [list(set(x)) for x in predictions]
predictions = [[y for y in x if y != 0 and y != 1] for x in predictions]

classes = pandas.read_csv("../../data/classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(0)[1].to_dict()

label_count = Counter([y for x in y_test for y in x]).most_common()
expected = {x[0]: x[1] for x in label_count}
report = {x: [0, 0, 0] for x in range(2, 104)}

matrix = confusion_matrix(y_test, predictions, report)
performance = evaluate(matrix)

file = open("rnn-performance.csv", 'w')
for k, v in report.items():
	tp, fp, fn = v[0], v[1], v[2]
	file.write("%s,%d,%d,%d,%d,%f,%f,%f\n" % (classes[k][2::], expected[k] if k in expected else 0, tp, fp, fn, 0, 0, 0))
file.close()

print(performance)
