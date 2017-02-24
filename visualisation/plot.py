import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import pickle

V = pickle.load(open('data_2d.pickle', 'rb'))
V = V[:10000]
transpose = list(zip(*V))
max_x = max(transpose[0])
min_x = min(transpose[0])
max_y = max(transpose[1])
min_y = min(transpose[1])
print("{} {} {} {}".format(max_x, min_x, max_y, min_y))

for v in V:
    plt.scatter(v[0], v[1], 0.5, color='grey')
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(())
plt.yticks(())
plt.title('Claims')
plt.show()
