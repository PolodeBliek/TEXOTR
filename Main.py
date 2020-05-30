import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import random
import copy
import os
#import itertools
#import pickle
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

def Main(file, LogOutput, Complex, Demo, PklOutput):
    s_0 = time.time()
    image = np.asarray(io.imread("ImExamples/" + file)) #img_as_float
    disp_img = img_as_float(io.imread("ImExamples/" + file))
    Table = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2])/255    
    if Demo:
        fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
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


    s_2a = time.time()
    VerSums3 = []
    VerSums7 = []
    VerSums8 = []
    HorSums3 = []
    HorSums7 = []
    HorSums8 = []
    VerSums4 = copy.deepcopy(VerSums2)
    HorSums4 = copy.deepcopy(HorSums2)
    
    Zeroes = [i for i, x in enumerate(HorSums2) if x == 0]
    Zeroes = list(np.ediff1d(Zeroes))
    Zeroes = list(set(Zeroes))
    Zeroes.sort()
    #Determining Charachterwidth by using the height of charachter lines from HorSums
    CharachterWidth = math.floor(0.75*Zeroes[1])
    Fac1 = 1.75 
    Fac2 = 1.50
    Fac3 = 1.25
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
            HStep1 = math.floor(Fac1 * len(SubList))
            HStep2 = math.floor(Fac2 * len(SubList))
            HStep3 = math.floor(Fac3 * len(SubList))
            HStep1 = 3 if HStep1 == 0 else HStep1
            HStep2 = 3 if HStep2 == 0 else HStep2
            HStep3 = 3 if HStep3 == 0 else HStep3
            for index in range(0, len(SubList) - HStep1, HStep1):
                Sum = 0
                for index2 in range(0, HStep1):
                    Sum += SubList[index + index2]
                HorSums3.append(Sum/HStep1)
            for index in range(0, len(SubList) - HStep2, HStep2):
                Sum = 0
                for index2 in range(0, HStep2):
                    Sum += SubList[index + index2]
                HorSums7.append(Sum/HStep2)
            for index in range(0, len(SubList) - HStep3, HStep3):
                Sum = 0
                for index2 in range(0, HStep3):
                    Sum += SubList[index + index2]
                HorSums8.append(Sum/HStep3)


    Peaks3 = []
    Peaks7 = []
    Peaks8 = []
    NPeaks7 = []
    if len(Columns) != 0:
        trail3 = len(VerSums3)
        trail7 = len(VerSums7)
        trail8 = len(VerSums8)
        VerSumsT = []
        if len(VerSums5[1]) < 1.5 * CharachterWidth:
            VerSums5 = VerSums5[1:]
        if len(VerSums5[-1]) < 1.5 * CharachterWidth:
            VerSums5 = VerSums5[:-1]
        for SubList in VerSums5:
            SubList3 = []
            SubList7 = []
            SubList8 = []
            VStep1 = math.floor(Fac1 * CharachterWidth)
            VStep2 = math.floor(Fac2 * CharachterWidth)
            VStep3 = math.floor(Fac3 * CharachterWidth)
            VStep1 = 3 if VStep1 == 0 else VStep1
            VStep2 = 3 if VStep2 == 0 else VStep2
            VStep3 = 3 if VStep3 == 0 else VStep3
            for index in range(0, len(SubList) - VStep1, VStep1):
                Sum = 0
                for index2 in range(0, VStep1):
                    Sum += SubList[index + index2]
                #VerSums3.append(Sum/VStep1)
                SubList3.append(Sum/VStep1)
            for index in range(0, len(SubList) - VStep2, VStep2):
                Sum = 0
                for index2 in range(0, VStep2):
                    Sum += SubList[index + index2]
                #VerSums7.append(Sum/VStep2)
                SubList7.append(Sum/VStep2)
            for index in range(0, len(SubList) - VStep3, VStep3):
                Sum = 0
                for index2 in range(0, VStep3):
                    Sum += SubList[index + index2]
                #VerSums8.append(Sum/VStep3)
                SubList8.append(Sum/VStep3)
           
            MinSubList3 = 0 if len(SubList3) == 0 else min(SubList3)
            MaxSubList3 = 0 if len(SubList3) == 0 else max(SubList3)
            MinSubList7 = 0 if len(SubList7) == 0 else min(SubList7)
            MaxSubList7 = 0 if len(SubList7) == 0 else max(SubList7)
            MinSubList8 = 0 if len(SubList8) == 0 else min(SubList8)
            MaxSubList8 = 0 if len(SubList8) == 0 else max(SubList8)
            SubList7.insert(0, MinSubList7)
            SubList7.append(MinSubList7)
            TNPeaks7 = []
            if len(SubList) < 2.5*CharachterWidth:
                TNPeaks7 += [math.floor(len(SubList7)/2), ]
            TPeaks3, _ = find_peaks(SubList3, height = MinSubList3 + 0.7*(MaxSubList3 - MinSubList3))
            TPeaks7, _ = find_peaks(SubList7, height = MinSubList7 + 0.5*(MaxSubList7 - MinSubList7))
            TPeaks8, _ = find_peaks(SubList8, height = MinSubList8 + 0.7*(MaxSubList8 - MinSubList8))
            TPeaks3 = list(TPeaks3)
            for index in range(len(TPeaks3)):
                TPeaks3[index] = TPeaks3[index] + trail3
            Peaks3 = Peaks3 + TPeaks3
            for index in range(len(TPeaks7)):
                TPeaks7[index] = TPeaks7[index] + trail7
            for index in range(len(TNPeaks7)):
                TNPeaks7[index] = TNPeaks7[index] + trail7
            NPeaks7 += TNPeaks7
            for index in range(len(TPeaks8)):
                TPeaks8[index] = TPeaks8[index] + trail8
            Peaks7 = Peaks7 + list(TPeaks7)
            Peaks8 = Peaks8 + list(TPeaks8)

            VerSums3 = VerSums3 + SubList3
            VerSums7 = VerSums7 + SubList7
            VerSums8 = VerSums8 + SubList8
            VerSumsT += [SubList7, ]
            trail3 += len(SubList3)
            trail7 += len(SubList7)
            trail8 += len(SubList8)
        # VerSums3.insert(0, min(VerSums3))
        VerSums7.insert(0, min(VerSums7))
        # VerSums8.insert(0, min(VerSums8))
        # VerSums3.append(min(VerSums3))
        VerSums7.append(min(VerSums7))
        # VerSums8.append(min(VerSums8))
        # for index in range(len(Peaks3)):
            # Peaks3[index] += 1
        # for index in range(len(Peaks7)):
            # Peaks7[index] += 1
        # for index in range(len(Peaks8)):
            # Peaks8[index] += 1
    else:
        VStep1 = math.floor(Fac1 * CharachterWidth)
        VStep2 = math.floor(Fac2 * CharachterWidth)
        VStep3 = math.floor(Fac3 * CharachterWidth)
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
        Peaks7, _ = find_peaks(VerSums7, min(VerSums7) + 0.5*(max(VerSums7) - min(VerSums7)))
        VerSums3.insert(0, min(VerSums3))
        VerSums7.insert(0, min(VerSums7))
        VerSums8.insert(0, min(VerSums8))
        VerSums3.append(min(VerSums3))
        VerSums7.append(min(VerSums7))
        VerSums8.append(min(VerSums8))






    
    s_2b = time.time()
    #Peaks2, _ = find_peaks(VerSums3, height = 0.5*max(VerSums3))
    
    Slice = True
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
        files = os.listdir("ImOutput")
        for fname in files:
            os.remove("ImOutput/" + fname)
        for index in range(0, len(Columns) - 1):
            for index2 in range(0, len(Rows) - 1):
                Box = patches.Rectangle((Columns[index], Rows[index2]), Columns[index + 1] - Columns[index], Rows[index2 + 1] - Rows[index2], edgecolor = "red", fill = False)
                ImBox = 255*Table[Rows[index2]:Rows[index2 + 1], Columns[index]:Columns[index + 1]]
                ax0.add_patch(Box)
                name = str(index) + "_" + str(index2)
                im = Image.fromarray(ImBox)
                im = im.convert('L')
                im.save("ImOutput/" + name+".png")
            # im = Image.fromarray(Box)
            # im = im.convert('RGB')


        ax0.set_xlabel(file)
        # ax2.plot(VerSums2, color = "black")
        # ax2.set_ylim(0, 2*max(VerSums2))
        # ax3.plot(VerSums7, color = "black")
        # ax3.set_ylim(0, 2*max(VerSums7))
        # for x in Peaks7:
        #     ax3.plot(x+1, VerSums7[x+1], "x", color = "red")
        # for x in NPeaks7:
        #     ax3.plot(x+1, VerSums7[x+1], "x", color = "blue")
        # XPeaks, _ = find_peaks(VerSums7, height = 0.5*(max(VerSums7) - min(VerSums7)) + min(VerSums7))
        # for x in XPeaks:
        #     ax3.plot(x, VerSums7[x], "x", color = "green")
        # for x in Peaks3:
        #     ax1.plot(x, VerSums3[x], "x", color = "red")
        # for x in Peaks7:
        #     ax2.plot(x, VerSums7[x], "x", color = "red")
        # for x in Peaks8:
        #     ax3.plot(x, VerSums8[x], "x", color = "red")
        # ax1.set_xlabel("VerSums")
        # ax2.set_xlabel("VerSums2")
        # ax3.set_xlabel("VerSums7")
        # NPeaks7, _ = find_peaks(VerSums7, height = 0.5*max(VerSums7))
        # for x in NPeaks7:
        #     ax3.plot(x, VerSums7[x], "x", color = "red")
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
    
    




    s_2c = time.time()
    s_3 = time.time()

