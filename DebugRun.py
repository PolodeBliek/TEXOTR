import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
x = electrocardiogram()[2000:4000]
peaks, _ = find_peaks(x, height=0)
print(peaks)
print(x)
print(peaks[0], x[peaks[0]])
plt.plot(x)
plt.plot(65, 0.705, "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()
