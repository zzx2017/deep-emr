import numpy
import pandas 
import glob
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

filenames = glob.glob("./data/training/*.txt")
dataframe = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in filenames])
dataset = dataframe.values

X = numpy.array([[int(y) for y in x.split(',')] for x in dataset[:, 0]])
Y = numpy.array([[int(y) for y in x.split(',')] for x in dataset[:, 1]])

nb_words = 30551
max_length = 922

X = sequence.pad_sequences(X, maxlen=max_length)
Y = sequence.pad_sequences(Y, maxlen=max_length)

Y = numpy.array([y for x in Y for y in x])
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_Y = np_utils.to_categorical(encoded_Y)
encoded_Y = numpy.array([encoded_Y[i:i + max_length] for i in range(0, len(encoded_Y), max_length)])

# print(encoded_Y[0].argmax(1))

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(nb_words, embedding_vector_length, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(40, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

print(model.summary())

model.fit(X, encoded_Y, epochs=10, batch_size=32)

model_json = model.to_json()
with open("rnn-model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("rnn-model.h5")
print("Saved model to disk")
