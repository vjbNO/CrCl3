import numpy as np
import matplotlib.pyplot as plt
import sys

dUp = np.loadtxt(sys.argv[1])
dDown = np.loadtxt(sys.argv[2])
d1 = np.subtract(dUp,dDown)
d2 = np.add(dUp,dDown)
lineAv1 = np.mean(d1,axis=1)
lineAv2 = np.mean(d2,axis=1)
plt.scatter(np.arange(len(lineAv1)),lineAv1,label='difference')
plt.scatter(np.arange(len(lineAv2)),lineAv2,label='sum')
plt.legend()
plt.show()
