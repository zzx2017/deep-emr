import numpy
import pandas 
import glob
import gensim
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from itertools import accumulate

dictionary = pandas.read_csv("./data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
training_files = glob.glob("./data/training/*.txt")
training = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in training_files]).values

X_train = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 0]])
Y_train = numpy.array([[int(y) for y in x.split(',')] for x in training[:, 1]])

empty_indices = numpy.array([i for i, x in enumerate(Y_train) if len(list(set(x))) == 1 and list(set(x))[0] == 1])

X_train = numpy.delete(X_train, empty_indices)
Y_train = numpy.delete(Y_train, empty_indices)

nb_words = 30551
max_length = 922

X_train = sequence.pad_sequences(X_train, maxlen=max_length)
Y_train = sequence.pad_sequences(Y_train, maxlen=max_length)

Y_train = numpy.array([y for x in Y_train for y in x])
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
encoded_Y = np_utils.to_categorical(encoded_Y)
encoded_Y = numpy.array([encoded_Y[i:i + max_length] for i in range(0, len(encoded_Y), max_length)])

embedding_vector_length = 100

word2vec_model = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
embedding_weights = numpy.zeros((nb_words, embedding_vector_length))

for word, index in dictionary.items():
	if word in word2vec_model:
		embedding_weights[index,:] = word2vec_model[word]

model = Sequential()
model.add(Embedding(nb_words, embedding_vector_length, input_length=max_length, mask_zero=True, weights=[embedding_weights]))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(40, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 

print(model.summary())
print(model.get_config())

model.fit(X_train, encoded_Y, epochs=20, batch_size=32)

model_json = model.to_json()
with open("lstm-word2vec-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("lstm-word2vec-model.h5")
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

test_files = glob.glob("./data/test/text/*.txt")
label_files = glob.glob("./data/test/labels/*.txt")
testset = [(pandas.read_csv(x, delim_whitespace=True, header=None)).values for x in test_files]
labels = pandas.concat([(pandas.read_csv(x, delim_whitespace=True, header=None)) for x in label_files]).values

X_test = [[[int(z) for z in str(y[0]).split(',')] for y in x] for x in testset]
Y_test = [[int(y) for y in str(x[0]).split(',')] for x in labels]
Y_test = [x if x[0] != 0 else [] for x in Y_test]

X_test = [sequence.pad_sequences(x, maxlen=max_length) for x in X_test]

json_file = open('lstm-word2vec-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("lstm-word2vec-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

predictions = list()
for record in X_test:
	prediction = loaded_model.predict(record)
	prediction = numpy.array([[[round(z) for z in y] for y in x] for x in prediction])
	prediction = [x.argmax(1) for x in prediction]
	prediction = [y for x in prediction for y in x]
	prediction = list(set([x for x in prediction]))
	prediction = [x for x in prediction if x != 0 and x != 1]
	predictions.append(prediction)

matrix = confusion_matrix(Y_test, predictions)
performance = evaluate(matrix)

print(matrix)
print(performance)
