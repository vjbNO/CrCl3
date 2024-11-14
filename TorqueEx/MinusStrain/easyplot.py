import numpy as np
import matplotlib.pyplot as plt
import sys

dUpP = np.loadtxt(sys.argv[1])
dDownP = np.loadtxt(sys.argv[2])
dUpM = np.loadtxt(sys.argv[3])
dDownM = np.loadtxt(sys.argv[4])
d = sys.argv[1:]

d1 = np.add(dUpP,dDownP)
d2 = np.add(dUpM,dDownM)
d = np.subtract(d2,d1)
lineAv1 = np.sum(d,axis=1)
#lineAv2 = np.sum(d2,axis=1)
plt.scatter(np.arange(len(lineAv1)),lineAv1,label='+')
#plt.scatter(np.arange(len(lineAv2)),lineAv2,label='-')
#plt.legend()
plt.show()
