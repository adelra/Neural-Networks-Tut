import numpy as np
from matplotlib import pylab

"""Q3.py: plotting SSE and Crossentropy."""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

sse_list = []
ce_list = []
Y = 0
range_of_floats = np.arange(0, 1, 0.01)
reversed_floats = range_of_floats[::-1]
for yhat in reversed_floats:
    CE = -np.sum(Y * np.log(yhat) + (1 - Y) * np.log(1 - yhat))
    SSE = np.sum((Y - yhat) ** 2)
    ce_list.append(CE)
    sse_list.append(SSE)
    print("CE", CE, "SSE", SSE)
    print()
pylab.plot(range_of_floats,ce_list, label="Cross_Entropy")
pylab.plot(range_of_floats,sse_list, label="SSE")
pylab.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)

pylab.show()
