

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pickle

data = pickle.load(open('large_clusters.pickle', 'rb'))
data = data[:1000]

reshaped = []
for row in data:
    reshaped.append([row[0][0], row[0][1], row[0][2], row[1]])
data = list(zip(*reshaped))

# annotate with color
color_map = {
    0: '#000000',
    1: '#000022',
    2: '#000044',
    3: '#000066',
    4: '#000088',
    5: '#002200',
    6: '#004400',
    7: '#006600',
    8: '#008800',
    9: '#220000',
    10: '#440000',
    11: '#660000',
    12: '#880000',
    12: '#0000AA',
    14: '#0000BB',
    15: '#00AA00',
    16: '#00BB00',
    17: '#AA0000',
    18: '#BB0000',
    19: '#0000CC',
    20: '#00CC00',
}

colorized_data = []
colorized_data.append(data[0])
colorized_data.append(data[1])
colorized_data.append(data[2])
colorized_data.append(list(map(lambda x:color_map[x], data[3])))

print(colorized_data[3][:10])

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.scatter(colorized_data[0], colorized_data[1], colorized_data[2], facecolors=colorized_data[3])
plt.ylim([-25, 25])
plt.xlim([-250, 250])

plt.show()
