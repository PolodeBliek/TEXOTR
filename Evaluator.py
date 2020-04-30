import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import random
import copy
import itertools
import pickle
import time

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

from Klad4 import Main
Outputs = []
Files = ["Example.png", "Example3.png", "Example4.png", "Example5.png", "Example6.png", "Example7.png", "Example8.png", "Example9.png", "Example10.png", "Example11.png", "Example12.png", "Example13.png", "Example14.png", "Example.png15", "Example16.png"]

Answers = ["02 08", "03 11", "02 10", "04 05", "02 05", "COMPLEX", "03 07", "03 04", "COMPLEX", "COMPLEX", "05 09", "03 06", "03 06", "02 02", "02 02"]
ConsoleOutput = True
Failures = []

t_0 = time.time()
print(Main("Example.png", False, True, False, False))
t_1 = time.time()
print("\n")
print(t_1 - t_0)

exit()


for file in Files:
    try:
        Output = Main(file, False, True, False, False)
    except:
        Output = "FAILURE"
    Outputs.append(Output)
    if ConsoleOutput:
        print(Output)
print("\n")
print("################")
for index in range(0, len(Files)):
    if Answers[index] == "COMPLEX":
        print(Files[index] + "\tTOO COMPLEX")
    else:
        if (Outputs[index][1] == int(Answers[index][0:2])) and (Outputs[index][2] == int(Answers[index][3:])) and (Outputs[index] != "FAIL"):
            Correct = True
        else:
            Correct = False
            Failures.append(index)
        print(Files[index] + "\t" + ("SUCCES" if Correct else "FAIL"))

print("\n")
print("###############")
print("FAILURES:")
print("\n")
for index in Failures:
    print("FILE: " + Files[index] + "\t OUTPUT: " + str(Outputs[index]) + ("\t CORRECT ANSWER: " if len(str(Outputs[index])) > 22 else "\t\t\t CORRECT ANSWER: ") + str(Answers[index]))








