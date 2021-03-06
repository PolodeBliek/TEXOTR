import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches
import random
import copy
import itertools
import pickle
import time
import tqdm

from scipy.ndimage import gaussian_filter
from scipy import signal
from skimage import data, io
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from skimage.exposure import histogram
import more_itertools as mit
import math


from Main import Main
Outputs = []
Files = ["Example.png", "Example3.png", "Example4.png", "Example5.png", "Example6.png", "Example8.png", "Example9.png", "Example10.png", "Example11.png", "Example13.png", "Example14.png", "Example15.png", "Example16.png"]
Answers = ["02 08", "03 11", "02 10", "04 05", "02 05", "03 07", "03 04", "03 00", "10 00", "03 06", "03 06", "02 02", "02 02"]
Failures = []
Errors = []
Times = []
t_begin = time.time()
for index in tqdm.tqdm(range(0, len(Files))):
    file = Files[index]
    try:
        Output = Main(file, False, True, False, False, True)
        Times.append(pickle.load(open("Times.pkl", "rb")))
        if not(Output == Answers[index][0:2]):
            Failures.append(file)
    except:
        Errors.append(file)
t_end = time.time()
print("SUCCES  :\t", len(Files)-len(Failures)-len(Errors))
print("FAILURES:\t", len(Failures))
for fail in Failures:
    print("\t", fail)
print("ERRORS  :\t", len(Errors))
for error in Errors:
    print("\t", error)
print("TOTAL TIME:\t", t_end - t_begin)
sys.stdout.write("\nSHOW TIMES? (y/n)")
sys.stdout.flush()
TimesAnswer = input()
TimesAnswer = True if TimesAnswer == "y" else False
if TimesAnswer:
    for element in Times:
        print(list(element))
    TimesResult = [0, ] * len(Times[0])
    for element in Times:
        for index in range(0, len(element)):
            TimesResult[index] += element[index]
    fig, (ax0) = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 4))
    ax0.set_xlabel("Times")
    ax0.plot(TimesResult)
    ax0.set_ylim(0, 1.5*max(TimesResult))
    plt.show()
    fig.tight_layout()

