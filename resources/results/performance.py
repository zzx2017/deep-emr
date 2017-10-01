import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl

mpl.style.use('classic')

plt.rc('text', usetex=True)
pl.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',    # helvetica font
    r'\usepackage{sansmath}',   # math-font matching  helvetica
    r'\sansmath'                # actually tell tex to use it!
    r'\usepackage{siunitx}',    # micro symbols
    r'\sisetup{detect-all}',    # force siunitx to use the fonts
]  

f1 = [0.9276, 0.9268, 0.9185, 0.9171, 0.9156, 0.9081, 0.8973, 0.8909, 0.8776, 0.8747, 0.8798, 0.8900, 0.9046, 0.9010, 0.9081]

models = np.array(['NLM', 'Harbin', 'Kaiser', 'Linguamatics', 'Nottingham', 'Ohio', 'TMUNSW', 'NCU', 'UNIMAN', 'Utah', 'CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM'])
model_positions = np.arange(len(models))

plt.bar(model_positions[0:10], f1[0:10], align='center', color = '0', linewidth=0.5)
plt.bar(model_positions[10::], f1[10::], align='center', color = '1', linewidth=0.5)
plt.xticks(model_positions, models, rotation = 'vertical')

plt.title('Comparison of System Performance')
plt.xlabel('Teams/Models')
plt.ylabel('F-measure')

plt.savefig('system-performance-1.pdf', format='pdf', bbox_inches='tight')
plt.close()

f1 = [0.8798, 0.8900, 0.9046, 0.9010, 0.9081, 0.815, 0.9276]

models = np.array(['CNN', 'RNN', 'GRU', 'LSTM', 'BLSTM', '2014 i2b2 (Mean)', '2014 i2b2 (Max)'])
model_positions = np.arange(len(models))

plt.bar(model_positions[0:5], f1[0:5], align='center', color = '1', linewidth=0.8)
plt.bar(model_positions[5::], f1[5::], align='center', color = '0', linewidth=0.8)
plt.xticks(model_positions, models, rotation = 'vertical')

plt.title('Comparison of System Performance')
plt.xlabel('Models')
plt.ylabel('F-measure')

plt.savefig('system-performance-2.pdf', format='pdf', bbox_inches='tight')
plt.close()