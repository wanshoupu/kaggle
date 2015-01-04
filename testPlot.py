import matplotlib.pyplot as plt
import numpy as np
import math

v = [[ 1.4,  7.3], [-0.4,  6.9], [-3.,   5.8], [-4.7,  4. ], [-7.3 , 1.3], [-6.5  ,0.9], [-6.6,  1.8], [-6.9 , 2.5], [-6.3 , 3.5], [-6.7 , 3.8]]

plt.subplot(221)
t = np.arange(0.01, 20.0, 0.01)
plt.semilogy(range(0,10), [math.exp(x) for x in range(0,10)])
plt.plot(v)
plt.show()
