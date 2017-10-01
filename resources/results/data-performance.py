import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pylab as pl

# mpl.style.use('classic')

plt.rc('text', usetex=True)
pl.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',    # helvetica font
    r'\usepackage{sansmath}',   # math-font matching  helvetica
    r'\sansmath'                # actually tell tex to use it!
    r'\usepackage{siunitx}',    # micro symbols
    r'\sisetup{detect-all}',    # force siunitx to use the fonts
] 

X = np.array([[762], [243], [79], [78], [1542], [88], [21], [409], [14], [993], [25], [8], [1538], [359], [948], [3], [283], [1271], [1391], [542], [314], [1], [36], [64], [629], [536], [20], [318], [1274], [462], [118], [281], [57], [9], [183], [148], [0], [21], [0]])
y = np.array([0.89010989, 0.733333333, 0.65, 0.283185841, 0.95567867, 0.557823129, 0, 0.910931174, 0.166666667, 0.957627119, 0.222222222, 0, 0.968622101, 0.637413395, 0.888527258, 0, 0.833787466, 0.930289944, 0.912993039, 0.891752577, 0.824742268, 0, 0.5, 0.823529412, 0.878751501, 0.857142857, 0.214285714, 0.774311927, 0.925858951, 0.93814433, 0.812030075, 0.877796902, 0.492753623, 0, 0.81938326, 0.811965812, 0.751515152, 0, 0.981169475])

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'black', s = 5, label = 'Risk factor indicator')
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black', linewidth = 0.8)
plt.title('Number of Training Instances and F-measure')
plt.xlabel('Number of Training Instances')
plt.ylabel('F-measure')
plt.legend()

plt.savefig('data-performance.pdf', format='pdf', bbox_inches='tight')
plt.close()
