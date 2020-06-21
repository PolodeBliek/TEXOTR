import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import random
import copy
import os
#import itertools
import pickle
import time
from PIL import Image

#from scipy.ndimage import gaussian_filter
#from scipy import signal
from skimage import data, io
from skimage import img_as_float
#from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from skimage.exposure import histogram
import more_itertools as mit
import math

def Main(file, LogOutput, Complex, Demo, PklOutput, timer = False):
    s = []
    s.append(time.time()) #0
    image = np.asarray(io.imread("ImExamples/" + file)) #img_as_float
    disp_img = img_as_float(io.imread("ImExamples/" + file))
    Table = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2])/255    
    if Demo:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    s.append(time.time()) #1
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
    for index in range(width):
        VerSlices.append(Table[:, index])
    for index in range(height):
        if max(HorSlices[index])-min(HorSlices[index])<0.1:
            EmptyHorSlices.append(index)
    for index in range(width):
        if max(VerSlices[index])-min(VerSlices[index])<0.1:
            EmptyVerSlices.append(index)
    s.append(time.time()) #2


    NegColor = np.amax(Table)
    
    VerSums = [1.0 - (1/height)*sum(x) for x in zip(*HorSlices)]
    HorSums = [1.0 - (1/width)*sum(x) for x in zip(*VerSlices)]    

    PotRowPeakHeight = max(HorSums)*0.5 if max(HorSums)*0.5 > 0.10 else 0.10
    PotColPeakHeight = max(VerSums)*0.5 if max(VerSums)*0.5 > 0.10 else 0.10

    PotCol, _ = find_peaks(VerSums, height = PotColPeakHeight)
    PotRow, _ = find_peaks(HorSums, height = PotRowPeakHeight)

    PotCol2 = []
    PotRow2 = []
    s.append(time.time()) #3

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
    s.append(time.time()) #4


    Columns = PotCol2
    Rows = PotRow2
    DroppedCols = list(set(PotColPeaks)-set(Columns))
    DroppedRows = list(set(PotRowPeaks)-set(Rows))

    VerSums2 = copy.deepcopy(VerSums)
    HorSums2 = copy.deepcopy(HorSums)
    ColumnWidth = 5 #On Both sides => Actual Width = ColumnWidth*2 + 1
    RowWidth = 5
    for x in Columns:
        VerSums2[x] = 0
        for x2 in range(ColumnWidth):
            try:
                VerSums2[x - x2] = 0
                VerSums2[x + x2] = 0
            except:
                pass
    for x in Rows:
        HorSums[x] = 0
        for x2 in range(RowWidth):
            try:
                HorSums2[x - x2] = 0
                HorSums2[x + x2] = 0
            except:
                pass
    s.append(time.time()) #5


    VerSums7 = []
    HorSums7 = []
    VerSums4 = copy.deepcopy(VerSums2)
    HorSums4 = copy.deepcopy(HorSums2)
    
    Zeroes = [i for i, x in enumerate(HorSums2) if x == 0]
    Zeroes = list(np.ediff1d(Zeroes))
    Zeroes = list(set(Zeroes))
    Zeroes.sort()
    #Determining Charachterwidth by using the height of charachter lines from HorSums
    CharachterWidth = math.floor(0.75*Zeroes[1])
    Fac2 = 1.50
    VerSums5 = []
    HorSums5 = []
    if len(Columns) != 0:
        VerSums5.append(VerSums4[:Columns[0]])
    for index in range(len(Columns)-1):
        VerSums5.append(VerSums4[Columns[index]:Columns[index+1]])
    for index in range(len(Rows) - 1):
        HorSums5.append(HorSums4[Rows[index]:Rows[index+1]])
    if len(Rows) != 0:
        HorSums5.append(HorSums4[Rows[-1]:])
    if len(Columns) != 0:
        VerSums5.append(VerSums4[Columns[-1]:])
    else:
        VerSums5 = copy.deepcopy(VerSums4)
    if len(Rows) != 0:
        for SubList in HorSums5:
            HStep2 = math.floor(Fac2 * len(SubList))
            HStep2 = 3 if HStep2 == 0 else HStep2
            for index in range(0, len(SubList) - HStep2, HStep2):
                Sum = 0
                for index2 in range(0, HStep2):
                    Sum += SubList[index + index2]
                HorSums7.append(Sum/HStep2)
    s.append(time.time()) #6


    Peaks7 = []
    NPeaks7 = []
    if len(Columns) != 0:
        trail7 = len(VerSums7)
        VerSumsT = []
        if len(VerSums5[1]) < 1.5 * CharachterWidth:
            VerSums5 = VerSums5[1:]
        if len(VerSums5[-1]) < 1.5 * CharachterWidth:
            VerSums5 = VerSums5[:-1]
        for SubList in VerSums5:
            SubList7 = []
            VStep2 = math.floor(Fac2 * CharachterWidth)
            VStep2 = 3 if VStep2 == 0 else VStep2
            for index in range(0, len(SubList) - VStep2, VStep2):
                Sum = 0
                for index2 in range(0, VStep2):
                    Sum += SubList[index + index2]
                SubList7.append(Sum/VStep2)
           
            MinSubList7 = 0 if len(SubList7) == 0 else min(SubList7)
            MaxSubList7 = 0 if len(SubList7) == 0 else max(SubList7)
            SubList7.insert(0, MinSubList7)
            SubList7.append(MinSubList7)
            TNPeaks7 = []
            if len(SubList) < 2.5*CharachterWidth:
                TNPeaks7 += [math.floor(len(SubList7)/2), ]
            TPeaks7, _ = find_peaks(SubList7, height = MinSubList7 + 0.5*(MaxSubList7 - MinSubList7))
            for index in range(len(TPeaks7)):
                TPeaks7[index] = TPeaks7[index] + trail7
            for index in range(len(TNPeaks7)):
                TNPeaks7[index] = TNPeaks7[index] + trail7
            NPeaks7 += TNPeaks7
            Peaks7 = Peaks7 + list(TPeaks7)

            VerSums7 = VerSums7 + SubList7
            VerSumsT += [SubList7, ]
            trail7 += len(SubList7)
        VerSums7.insert(0, min(VerSums7))
        VerSums7.append(min(VerSums7))
    else:
        VStep2 = math.floor(Fac2 * CharachterWidth)
        VStep2 = 3 if VStep2 == 0 else VStep2
        for index in range(0, len(VerSums5) - VStep2, VStep2):
            Sum = 0
            for index2 in range(0, VStep2):
                Sum += VerSums5[index + index2]
            VerSums7.append(Sum/VStep2)
        Peaks7, _ = find_peaks(VerSums7, min(VerSums7) + 0.5*(max(VerSums7) - min(VerSums7)))
        VerSums7.insert(0, min(VerSums7))
        VerSums7.append(min(VerSums7))
    s.append(time.time()) #7






    
    #Peaks2, _ = find_peaks(VerSums3, height = 0.5*max(VerSums3))
    
    Slice = True
    NPeaks7, _ = find_peaks(VerSums7, height = 0.5*max(VerSums7))
    if Demo:
        rect2 = patches.Rectangle((200, 15), 40, 30, linewidth = 1, edgecolor = "red", fill = False)
        rect = patches.Rectangle((20, 15), 40, 30, linewidth = 1, edgecolor = 'r', fill = False)
        ax0.imshow(disp_img, vmin=image.min(), vmax=image.max(), cmap='gray')
        # ax0.add_patch(rect)
        # ax0.add_patch(rect2)
        # ax1.plot(VerSums, color = "black")
        # ax1.set_ylim(0, 2*max(VerSums))
        # for index in range(0, len(Rows)):
        #      ax0.axhline(Rows[index], color="r")
        # for index1 in range(0, len(Columns)):
        #     ax0.axvline(Columns[index1], color="r")
        # for index in range(0, len(Rows)):
        #     for index2 in range(0, len(Columns)):
        #         ax0.plot(Columns[index2], Rows[index], "x", color = "red")
        # files = os.listdir("ImOutput")
        # for fname in files:
        #     os.remove("ImOutput/" + fname)
        # for index in range(0, len(Columns) - 1):
        #     for index2 in range(0, len(Rows) - 1):
        #         Box = patches.Rectangle((Columns[index], Rows[index2]), Columns[index + 1] - Columns[index], Rows[index2 + 1] - Rows[index2], edgecolor = "red", fill = False)
        #         ImBox = 255*Table[Rows[index2]:Rows[index2 + 1], Columns[index]:Columns[index + 1]]
        #         ax0.add_patch(Box)
        #         name = str(index) + "_" + str(index2)
        #         im = Image.fromarray(ImBox)
        #         im = im.convert('L')
        #         im.save("ImOutput/" + name+".png")


        ax0.set_xlabel(file)
        ax1.plot(VerSums7, color = "black")
        ax1.set_ylim(0, 2*max(VerSums7))
        XPeaks, _ = find_peaks(VerSums7, height = 0.5*(max(VerSums7) - min(VerSums7)) + min(VerSums7))
        ax1.set_xlabel("VerSums")
        for x in NPeaks7:
            ax1.plot(x, VerSums7[x], "x", color = "blue")
        #axb.set_ylim(0, 2*max(VerSums2))
        ##Rows2 = Rows[1:]
        ##for index in Rows2:
        ##    axb.axvline(index - Rows[0], color = "red")
        #ax2.plot(VerSums7, color = "black")
        ##ax2.title("FACTOR: " + str(Fac2) + " STEP: " + str(VStep2))
        #ax2.set_ylim(0, 2*max(VerSums7))
        #ax3.plot(VerSums8, color = "black")
        #ax3.set_ylim(0, 2*max(VerSums8))
        #ax3.title("FACTOR: " + str(Fac3) + " STEP: " + str(VStep3))       
        #for index in Peaks2:
        #    ax1.plot(index, VerSums3[index], "x", color = "red")
    if Demo:
        plt.show()
        fig.tight_layout()
    
    if timer:
        pickle.dump(np.diff(s), open("Times.pkl", "wb+"))
    Output = "%02d" % len(NPeaks7) 
    return Output
