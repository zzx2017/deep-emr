import numpy
import pandas 
import glob
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json

filenames = glob.glob("./data/training/*.txt")
dataframe = pandas.concat([pandas.read_csv(filename, delim_whitespace=True, header=None) for filename in filenames])
dataset = dataframe.values

X = numpy.array([[int(y) for y in x.split(',')] for x in dataset[:, 0]])
Y = numpy.array([[int(y) for y in str(x).split(',')] if str(x) != '0' else [] for x in dataset[:, 1]])

empty_indices = numpy.array([i for i, x in enumerate(Y) if x == []])

X_train = numpy.delete(X, empty_indices)
Y_train = numpy.delete(Y, empty_indices)

mlb = MultiLabelBinarizer()
encoded_Y = mlb.fit_transform(Y_train)

vocabulary = 30550
max_length = 922

X = sequence.pad_sequences(X, maxlen=max_length)
X_train = sequence.pad_sequences(X_train, maxlen=max_length)

model = Sequential()
model.add(Embedding(vocabulary, 32, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(254, activation='relu'))
model.add(Dense(38, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, encoded_Y, validation_data=(X_train, encoded_Y), epochs=60, batch_size=32)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = loaded_model.predict(X[0:32])
rounded = numpy.array([[round(y) for y in x] for x in predictions])

predicted_Y = mlb.inverse_transform(rounded)

for result in predicted_Y:
	print(result)
