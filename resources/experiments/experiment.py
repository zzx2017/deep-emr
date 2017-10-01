import pandas
import numpy as np
import matplotlib.pyplot as plt

# Using Hyperopt to estimate the hyperparameter values results in the model performance in these ranges.
# The experiments were conducted by running Hyperopt multiple times and trying out those estimated parameters.

cnn = pandas.read_csv('cnn-experiment.csv', header=None).values

cnn_r = cnn[:, 0]
cnn_p = cnn[:, 1]
cnn_f = cnn[:, 2]

X = np.arange(len(cnn_f))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, cnn_r, label = 'Recall')
plt.plot(X, cnn_p, label = 'Precision')
plt.plot(X, cnn_f, label = 'F1')

plt.title('CNN Experiments')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('Score')
plt.legend()

plt.savefig('cnn-experiments.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([cnn_r, cnn_p, cnn_f])
plt.xticks([1, 2, 3], ['Recall', 'Precision', 'F1'])
plt.title('CNN Performance')
plt.xlabel('Evaluation Measures')
plt.ylabel('Score')

plt.savefig('cnn-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

rnn = pandas.read_csv('rnn-experiment.csv', header=None).values

rnn_r = rnn[:, 0]
rnn_p = rnn[:, 1]
rnn_f = rnn[:, 2]

X = np.arange(len(rnn_f))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, rnn_r, label = 'Recall')
plt.plot(X, rnn_p, label = 'Precision')
plt.plot(X, rnn_f, label = 'F1')

plt.title('RNN Experiments')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('Score')
plt.legend()

plt.savefig('rnn-experiments.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([rnn_r, rnn_p, rnn_f])
plt.xticks([1, 2, 3], ['Recall', 'Precision', 'F1'])
plt.title('RNN Performance')
plt.xlabel('Evaluation Measures')
plt.ylabel('Score')

plt.savefig('rnn-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

gru = pandas.read_csv('gru-experiment.csv', header=None).values

gru_r = gru[:, 0]
gru_p = gru[:, 1]
gru_f = gru[:, 2]

X = np.arange(len(gru_f))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, gru_r, label = 'Recall')
plt.plot(X, gru_p, label = 'Precision')
plt.plot(X, gru_f, label = 'F1')

plt.title('GRU Experiments')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('Score')
plt.legend()

plt.savefig('gru-experiments.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([gru_r, gru_p, gru_f])
plt.xticks([1, 2, 3], ['Recall', 'Precision', 'F1'])
plt.title('GRU Performance')
plt.xlabel('Evaluation Measures')
plt.ylabel('Score')

plt.savefig('gru-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

lstm = pandas.read_csv('lstm-experiment.csv', header=None).values

lstm_r = lstm[:, 0]
lstm_p = lstm[:, 1]
lstm_f = lstm[:, 2]

X = np.arange(len(lstm_f))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, lstm_r, label = 'Recall')
plt.plot(X, lstm_p, label = 'Precision')
plt.plot(X, lstm_f, label = 'F1')

plt.title('LSTM Experiments')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('Score')
plt.legend()

plt.savefig('lstm-experiments.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([lstm_r, lstm_p, lstm_f])
plt.xticks([1, 2, 3], ['Recall', 'Precision', 'F1'])
plt.title('LSTM Performance')
plt.xlabel('Evaluation Measures')
plt.ylabel('Score')

plt.savefig('lstm-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

blstm = pandas.read_csv('blstm-experiment.csv', header=None).values

blstm_r = blstm[:, 0]
blstm_p = blstm[:, 1]
blstm_f = blstm[:, 2]

X = np.arange(len(blstm_f))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, blstm_r, label = 'Recall')
plt.plot(X, blstm_p, label = 'Precision')
plt.plot(X, blstm_f, label = 'F1')

plt.title('BLSTM Experiments')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('Score')
plt.legend()

plt.savefig('blstm-experiments.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([blstm_r, blstm_p, blstm_f])
plt.xticks([1, 2, 3], ['Recall', 'Precision', 'F1'])
plt.title('BLSTM Performance')
plt.xlabel('Evaluation Measures')
plt.ylabel('Score')

plt.savefig('blstm-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(X, cnn_f, label = 'CNN')
plt.plot(X, rnn_f, label = 'RNN')
plt.plot(X, gru_f, label = 'GRU')
plt.plot(X, lstm_f, label = 'LSTM')
plt.plot(X, blstm_f, label = 'BLSTM')
plt.title('Performance of Different Deep Learning Models')
plt.xlabel('Number of Experimental Trials')
plt.ylabel('F1')
plt.legend()

plt.savefig('experiment-comparison.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.boxplot([cnn_f, rnn_f, gru_f, lstm_f, blstm_f])
plt.xticks([1, 2, 3, 4, 5], ['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
plt.title('Performance of Different Deep Learning Models')
plt.xlabel('Models')
plt.ylabel('F1')

plt.savefig('performance-comparison.pdf', format='pdf', bbox_inches='tight')
plt.close()
