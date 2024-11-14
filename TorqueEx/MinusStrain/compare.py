import numpy as np
import matplotlib.pyplot as plt
import sys

data = sys.argv[1:]
for d in data:
    dat = np.loadtxt(d)
    line = np.sum(dat,axis=1)
    plt.scatter(np.arange(len(line)),line,label=d)
    plt.legend()
plt.show()
