import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
#import random
import copy
#import itertools
#import pickle
import time

#from scipy.ndimage import gaussian_filter
#from scipy import signal
from skimage import data, io
from skimage import img_as_float
#from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from skimage.exposure import histogram
#import more_itertools as mit
import math

def Main(file, LogOutput, Complex, Demo, PklOutput):
    s_0 = time.time()
    image = np.asarray(io.imread("ImExamples/" + file)) #img_as_float
    disp_img = img_as_float(io.imread("ImExamples/" + file))
    Table = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2])/255    
    if Demo:
        fig, (ax0, axb, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    s_1 = time.time()
    height, width = Table.shape
    height = height
    width = width

    

    HorSlices = []
    VerSlices = []
    EmptyHorSlices = []
    EmptyVerSlices = []
    AllPixels = []

    for index in range(height):
        HorSlices.append(list(Table[index]))
    s_1a = time.time()
    for index in range(width):
        VerSlices.append(Table[:, index])
    s_1b = time.time()

    for index in range(height):
        if max(HorSlices[index])-min(HorSlices[index])<0.1:
            EmptyHorSlices.append(index)
    s_1c = time.time()
    for index in range(width):
        if max(VerSlices[index])-min(VerSlices[index])<0.1:
            EmptyVerSlices.append(index)
    s_2 = time.time()


    NegColor = np.amax(Table)
    
    VerSums = [1.0 - (1/height)*sum(x) for x in zip(*HorSlices)]
    HorSums = [1.0 - (1/width)*sum(x) for x in zip(*VerSlices)]    

    PotRowPeakHeight = max(HorSums)*0.5 if max(HorSums)*0.5 > 0.10 else 0.10
    PotColPeakHeight = max(VerSums)*0.5 if max(VerSums)*0.5 > 0.10 else 0.10

    PotCol, _ = find_peaks(VerSums, height = PotColPeakHeight)
    PotRow, _ = find_peaks(HorSums, height = PotRowPeakHeight)

    PotCol2 = []
    PotRow2 = []

    for index in range(len(PotCol)):
        PotColPeaks, _ = find_peaks(VerSlices[PotCol[index]], height = min(VerSlices[PotCol[index]]) + (max(VerSlices[PotCol[index]])-min(VerSlices[PotCol[index]]))*0.5)
        PotColPeaks = list(set(PotColPeaks)-set(EmptyHorSlices)-set(PotRow))
        if len(PotColPeaks) == 0:
            PotCol2.append(PotCol[index])
        else:
            pass


    for index in range(len(PotRow)):
        PotRowPeaks, _ = find_peaks(HorSlices[PotRow[index]], height = min(HorSlices[PotRow[index]]) + (max(HorSlices[PotRow[index]])-min(HorSlices[PotRow[index]]))*0.5)
        PotRowPeaks = list(set(PotRowPeaks)-set(EmptyVerSlices)-set(PotCol))
        if len(PotRowPeaks) == 0:
            PotRow2.append(PotRow[index])
        else:
            pass


    Columns = PotCol2
    Rows = PotRow2
    DroppedCols = list(set(PotColPeaks)-set(Columns))
    DroppedRows = list(set(PotRowPeaks)-set(Rows))

    VerSums2 = copy.deepcopy(VerSums)
    ColumnWidth = 5 #On Both sides => Actual Width = ColumnWidth*2 + 1
    for x in Columns:
        VerSums2[x] = 0
        for x2 in range(ColumnWidth):
            VerSums2[x - x2] = 0
            VerSums2[x + x2] = 0


    s_2a = time.time()
    VerSums3 = []
    VerSums7 = []
    VerSums8 = []
    HorSums3 = []
    VerSums4 = copy.deepcopy(VerSums2)
    Fac1 = 0.22
    Fac2 = 0.15
    Fac3 = 0.05
    VerSums5 = []
    #print("VStep: ", VStep)
    #for index in range(0, len(VerSums2)-VStep, VStep):
    #    Sum = 0
    #    for index2 in range(0, VStep):
    #        Sum += VerSums2[index + index2]
    #    VerSums3.append(Sum/VStep)
    for index in range(len(Columns)-1):
        VerSums5.append(VerSums4[Columns[index]:Columns[index+1]])
    if len(Columns) != 0:
        VerSums5.append(VerSums4[Columns[-1]:])
    else:
        VerSums5 = copy.deepcopy(VerSums4)
    if len(Columns) != 0:
        for SubList in VerSums5:
            VStep1 = math.floor(Fac1 * len(SubList))
            VStep2 = math.floor(Fac2 * len(SubList))
            VStep3 = math.floor(Fac3 * len(SubList))
            VStep1 = 3 if VStep1 == 0 else VStep1
            VStep2 = 3 if VStep2 == 0 else VStep2
            VStep3 = 3 if VStep3 == 0 else VStep3
            for index in range(0, len(SubList) - VStep1, VStep1):
                Sum = 0
                for index2 in range(0, VStep1):
                    Sum += SubList[index + index2]
                VerSums3.append(Sum/VStep1)
            for index in range(0, len(SubList) - VStep2, VStep2):
                Sum = 0
                for index2 in range(0, VStep2):
                    Sum += SubList[index + index2]
                VerSums7.append(Sum/VStep2)
            for index in range(0, len(SubList) - VStep3, VStep3):
                Sum = 0
                for index2 in range(0, VStep3):
                    Sum += SubList[index + index2]
                VerSums8.append(Sum/VStep3)
        VerSums3.insert(0, min(VerSums3))
        VerSums7.insert(0, min(VerSums7))
        VerSums8.insert(0, min(VerSums8))
        VerSums3.append(min(VerSums3))
        VerSums7.append(min(VerSums7))
        VerSums8.append(min(VerSums8))
    else:
        VStep1 = math.floor(Fac1 * len(VerSums5))
        VStep2 = math.floor(Fac2 * len(VerSums5))
        VStep3 = math.floor(Fac3 * len(VerSums5))
        VStep1 = 3 if VStep1 == 0 else VStep1
        VStep2 = 3 if VStep2 == 0 else VStep2
        VStep3 = 3 if VStep3 == 0 else VStep3
        for index in range(0, len(VerSums5) - VStep1, VStep1):
            Sum = 0
            for index2 in range(0, VStep1):
                Sum += VerSums5[index + index2]
            VerSums3.append(Sum/VStep1)
        for index in range(0, len(VerSums5) - VStep2, VStep2):
            Sum = 0
            for index2 in range(0, VStep2):
                Sum += VerSums5[index + index2]
            VerSums7.append(Sum/VStep2)
        for index in range(0, len(VerSums5) - VStep3, VStep3):
            Sum = 0
            for index2 in range(0, VStep3):
                Sum += VerSums5[index + index2]
            VerSums8.append(Sum/VStep3)
        VerSums3.insert(0, min(VerSums3))
        VerSums7.insert(0, min(VerSums7))
        VerSums8.insert(0, min(VerSums8))
        VerSums3.append(min(VerSums3))
        VerSums7.append(min(VerSums7))
        VerSums8.append(min(VerSums8))






    s_2b = time.time()
    #Peaks2, _ = find_peaks(VerSums3, height = 0.5*max(VerSums3))
    
    if Demo:
        ax0.imshow(disp_img, vmin=image.min(), vmax=image.max(), cmap='gray')
        for index in range(0, len(Rows)):
             ax0.axhline(Rows[index], color="r")
        for index1 in range(0, len(Columns)):
            ax0.axvline(Columns[index1], color="r")

        VerSums6 = []
        ax1.plot(VerSums3, color = "black", label = "Vertical Fill")
        ax1.set_ylim(0, 2*max(VerSums3))
        Peaks3, _ = find_peaks(VerSums3, height = min(VerSums3) + 0.5*(max(VerSums3) - min(VerSums3)))
        for x in Peaks3:
            ax1.plot(x, VerSums3[x], "x", color = "red")
        Peaks7, _ = find_peaks(VerSums7, height = min(VerSums7) + 0.5*(max(VerSums7) - min(VerSums7)))
        Peaks8, _ = find_peaks(VerSums8, height = min(VerSums8) + 0.5*(max(VerSums8) - min(VerSums8)))
        for x in Peaks7:
            ax2.plot(x, VerSums7[x], "x", color = "red")
        for x in Peaks8:
            ax3.plot(x, VerSums8[x], "x", color = "red")
        axb.plot(VerSums2, color = "black", label = "BASELINE")
        axb.set_ylim(0, 2*max(VerSums2))
        Columns2 = Columns[1:]
        for index in Columns2:
            axb.axvline(index - Columns[0], color = "red")
        ax2.plot(VerSums7, color = "black")
        #ax2.title("FACTOR: " + str(Fac2) + " STEP: " + str(VStep2))
        ax2.set_ylim(0, 2*max(VerSums7))
        ax3.plot(VerSums8, color = "black")
        ax3.set_ylim(0, 2*max(VerSums8))
        #ax3.title("FACTOR: " + str(Fac3) + " STEP: " + str(VStep3))       
        #for index in Peaks2:
        #    ax1.plot(index, VerSums3[index], "x", color = "red")
    if Demo:
        plt.show()
        fig.tight_layout()
    
    




    s_2c = time.time()
    s_3 = time.time()

