import numpy as np
import sys
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

#Main(file, LogOutput, Complex, Demo, PklOutput)
from Main import Main
Outputs = []
Files = ["Example.png", "Example3.png", "Example4.png", "Example5.png", "Example6.png", "Example7.png", "Example8.png", "Example9.png", "Example10.png", "Example11.png", "Example12.png", "Example13.png", "Example14.png", "Example15.png", "Example16.png"]

FilestoRun = Files[0]

Answers = ["02 08", "03 11", "02 10", "04 05", "02 05", "04 00", "03 07", "03 04", "03 00", "10 00", "05 09", "03 06", "03 06", "02 02", "02 02"]
CompleteEvaluation = False
ConsoleOutput = True
Failures = []

if type(FilestoRun) == str:
    FilestoRun = [FilestoRun, ]
print(FilestoRun)

if len(FilestoRun) == 1:
    Output = Main(FilestoRun[0],  False, True, True, False)
else:
    sys.stdout.write("Show Results? (y/n)")
    sys.stdout.flush()
    showResult = input()
    showResult = True if showResult == "y" else False
    for file in FilestoRun:
        #sys.stdout.write("\n" + str(file) +  "\tPROCESSING")
        #sys.stdout.flush()
        try:
            t_begin = time.time()
            Output = Main(file, False, True, showResult, False, True)
            t_end = time.time()
            print("SNATCHED FROM PKL: ", list(pickle.load(open("Times.pkl", "rb"))))
            print("\n")
            if Output == Answers[Files.index(file)][0:2]:
                result = "SUCCES"
            elif Answers[Files.index(file)] == "COMPLEX":
                result = "COMPLEX"
            else:
                result = "FAILURE"
            #sys.stdout.write("\r" + str(file) + "\t" + result + "    ")
            #sys.stdout.flush()
            #if result == "SUCCES":
                #sys.stdout.write("\t" + str(t_end - t_begin))
                #sys.stdout.flush()
            #if result == "FAILURE":
                #sys.stdout.write("\t" + Output + " vs " + Answers[Files.index(file)])
                #sys.stdout.flush()
        except:
            sys.stdout.write("\r" + str(file) + "\tERROR    ")
            sys.stdout.flush()
# exit()
# for file in FilestoRun:
#     if len(FilestoRun) == 1:
#         Output = Main(file, False, True, True, False)
#     else:
#         print("Show Results? (y/n)")
#         showResult = input()
#         showResult = True if showResult == "y" else False

#     except:
#         Output = "FAILURE"
#     Outputs.append(Output)
#     if ConsoleOutput:
#         pass
#         # print(Output)
# if CompleteEvaluation:
#     print("\n")
#     print("################")
#     for index in range(0, len(Files)):
#         if Answers[index] == "COMPLEX":
#             print(Files[index] + "\tTOO COMPLEX")
#         else:
#             if (Outputs[index][1] == int(Answers[index][0:2])) and (Outputs[index][2] == int(Answers[index][3:])) and (Outputs[index] != "FAIL"):
#                 Correct = True
#             else:
#                 Correct = False
#                 Failures.append(index)
#             print(Files[index] + "\t" + ("SUCCES" if Correct else "FAIL"))

#     print("\n")
#     print("###############")
#     print("FAILURES:")
#     print("\n")
#     for index in Failures:
#         print("FILE: " + Files[index] + "\t OUTPUT: " + str(Outputs[index]) + ("\t CORRECT ANSWER: " if len(str(Outputs[index])) > 22 else "\t\t\t CORRECT ANSWER: ") + str(Answers[index]))









