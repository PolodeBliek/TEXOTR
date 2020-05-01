import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import random
import copy
import itertools
import pickle

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
import time


def avg(l):
    #Function Used to determine the average of a given list
    return (sum(l)/len(l))

def Distance(Number, List):
    #Function used to find the distance of a number to the nearest element in that list
    ListTemp = copy.deepcopy(List)
    ListTemp = [abs(x-Number) for x in ListTemp]
    return (min(ListTemp))


def intersection(lst1, lst2):
    #Intersection of 2 lists
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

LogOutput   = True
Complex     = True
Demo        = True
PklOutput   = True
file = 'Example8.png'




def Main(file, LogOutput, Complex, Demo, PklOutput):
    s_0 = time.time()
    image = np.asarray(io.imread("ImExamples/" + file)) #img_as_float
    disp_img = img_as_float(io.imread("ImExamples/" + file))
    Table = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2])/255    
    if Demo:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    s_1 = time.time()
    height, width = Table.shape
    height = height
    width = width

    

    HorSlices = []
    VerSlices = []
    EmptyHorSlices = []
    EmptyVerSlices = []
    AllPixels = []

    #for index in range(width):
    #    VerSlices.append([])
    #for index in range(height):
    #    HorSlices.append(Table[index].tolist())
    #    AllPixels = AllPixels + Table[index].tolist()
    #    for index2 in range(width):
    #        VerSlices[index2].append(Table[index, index2])
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

    NeutVerSums = [-sum(x) for x in zip(*HorSlices)]
    NeutHorSums = [-sum(x) for x in zip(*VerSlices)]
    NeutVerSums2 = [x-min(NeutVerSums) for x in NeutVerSums]
    VerSums = [1.0 - (1/height)*sum(x) for x in zip(*HorSlices)]
    HorSums = [1.0 - (1/width)*sum(x) for x in zip(*VerSlices)]
    s_2a = time.time()
    VerSums3 = []
    HorSums3 = []
    VStep = math.floor(width/20)
    for index in range(0, len(VerSums)-VStep, VStep):
        Sum = 0
        for index2 in range(0, VStep):
            Sum += VerSums[index + index2]
        VerSums3.append(Sum/VStep)
    HStep = math.floor(height/20)
    for index in range(0, len(HorSums)-HStep, HStep):
        Sum = 0
        for index2 in range(0, HStep):
            Sum += HorSums[index + index2]
        HorSums3.append(Sum/HStep)
    s_2b = time.time()
    CutOff = 0.3 * max(VerSums3)
    VerSums4 = np.asarray(copy.deepcopy(VerSums3)) < CutOff
    VerSums4 = list(VerSums4)
    VerSums4 = list(map(int, VerSums4))
    VerSums4 = list(np.ediff1d(VerSums4))
    HCutOff = 0.2 * max(VerSums3)
    HorSums4 = np.asarray(copy.deepcopy(HorSums3)) < HCutOff
    HorSums4 = list(HorSums4)
    HorSums4 = list(map(int, HorSums4))
    HorSums4 = list(np.ediff1d(HorSums4))
    Peaks2, _ = find_peaks(VerSums3, height = 0.5*max(VerSums3))



    s_2c = time.time()
    print(VerSums4)
    print(VerSums4.count(-1) + 1)

    s_3 = time.time()




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

    EmptyHorSlices2 = []
    EmptyVerSlices2 = []
    s_4 = time.time()

    if len(Rows) != 0:
        for index in range(len(EmptyHorSlices)):
            dist = [abs(i-EmptyHorSlices[index]) for i in Rows]
            if min(dist) <= 10:
                pass
            else:
                EmptyHorSlices2.append(EmptyHorSlices[index])

    if len(Columns) != 0:
        for index in range(len(EmptyVerSlices)):
            dist = [abs(i-EmptyVerSlices[index]) for i in Columns]
            if min(dist) <= 10:
                pass
            else:
                EmptyVerSlices2.append(EmptyVerSlices[index])

    NegRows = [list(group) for group in mit.consecutive_groups(EmptyHorSlices2)]
    NegCols = [list(group) for group in mit.consecutive_groups(EmptyVerSlices2)]
    NegCols = [int(sum(group)/len(group)) for group in NegCols]
    NegRows = [int(sum(group)/len(group)) for group in NegRows]

    if Complex:
        VerAvg = avg(VerSums)
        VerSeperators = list(set(Columns + NegCols + [width, 0]))
        VerSeperators.sort()
        TextColumns= []
        VerSumsPieces = [None for x in range(len(VerSeperators))]
        for index in range(len(VerSeperators)-1):
            if VerSeperators[index+1]- 5 - (VerSeperators[index]+5) >= 1:
                VerSumsPieces[index] = VerSums[VerSeperators[index]+5:VerSeperators[index+1]-5]
        for element in VerSumsPieces:
            if type(element)==list and len(element) != 0:
                TextColumns.append(element)
        for index in range(len(VerSums)):
            if Distance(index, VerSeperators) <= 5:
                VerSums[index] = 0
        Boundaries = []
        for index in range(len(VerSeperators)-1):
            Boundaries.append([VerSeperators[index],VerSeperators[index + 1]])
        Boundaries2 = []
        for index in range(len(Boundaries)):
            if max(VerSums[Boundaries[index][0]:Boundaries[index][1]]) >= 0.5*VerAvg:
                Boundaries2.append(Boundaries[index])
        SymmetryFactors = []
        for index in range(len(Boundaries2)):
            SymmetryList = []
            for index2 in range(Boundaries2[index][1]-Boundaries2[index][0]):
                SymmetryList.append(abs(sum(VerSums[Boundaries2[index][0]:index2+Boundaries2[index][0]])/sum(VerSums[Boundaries2[index][0]:Boundaries2[index][1]])-0.5))
            MidPoint = SymmetryList.index(min(SymmetryList)) + Boundaries2[index][0]
            SymmetryFactor = ((MidPoint-Boundaries2[index][0])/(Boundaries2[index][1]-Boundaries2[index][0])-0.5)
            if abs(SymmetryFactor)<=0.10:
                SymmetryFactors.append("C")
            elif SymmetryFactor<=-0.10:
                SymmetryFactors.append("L")
            else:
                SymmetryFactors.append("R")

        HorAvg = avg(HorSums)
        HorSeperators = list(set(Rows + NegRows + [height, 0]))
        HorSeperators.sort()
        TextColumns= []
        HorSumsPieces = [None for x in range(len(HorSeperators))]
        for index in range(len(HorSeperators)-1):
            if HorSeperators[index+1]- 5 - (HorSeperators[index]+5) >= 1:
                HorSumsPieces[index] = VerSums[HorSeperators[index]+5:HorSeperators[index+1]-5]
        for element in HorSumsPieces:
            if type(element)==list and len(element) != 0:
                TextColumns.append(element)
        for index in range(len(HorSums)):
            if Distance(index, HorSeperators) <= 5:
                HorSums[index] = 0
        HorBoundaries = []
        for index in range(len(HorSeperators)-1):
            HorBoundaries.append([HorSeperators[index],HorSeperators[index + 1]])
        HorBoundaries2 = []
        for index in range(len(HorBoundaries)):
            if max(HorSums[HorBoundaries[index][0]:HorBoundaries[index][1]]) >= 0.5*HorAvg:
                HorBoundaries2.append(HorBoundaries[index])
        TextRows = [(group[1]-group[0])/2 + group[0] for group in HorBoundaries2]
        TextLocs = [(group[1]-group[0])/2 + group[0] for group in Boundaries2]
        HorPieces = Rows + NegRows + TextRows
        HorPieces.sort()
        for index in range(len(HorPieces)):
            if HorPieces[index] in Rows:
                HorPieces[index] = "R"
            if HorPieces[index] in NegRows:
                HorPieces[index] = "N"
            if HorPieces[index] in TextRows:
                HorPieces[index] = "T"
        HorPieces = HorPieces[1:] if HorPieces[0] == "N" else HorPieces
        HorPieces = HorPieces[:-1] if HorPieces[-1] == "N" else HorPieces

        VerPieces = Columns + NegCols + TextLocs
        VerPieces.sort()
        for index in range(len(VerPieces)):
            if VerPieces[index] in Columns:
                VerPieces[index] = "C"
            if VerPieces[index] in NegCols:
                VerPieces[index] = "N"
            if VerPieces[index] in TextLocs:
                VerPieces[index] = "T"
        VerPieces = VerPieces[1:] if VerPieces[0] == "N" else VerPieces
        VerPieces = VerPieces[:-1] if VerPieces[-1] == "N" else VerPieces
        Row = [[] for index in range(len(TextLocs))]
        Row2 = [[] for index in range(len(TextLocs))]
        Markers = []
        for index in range(len(Row)):
            Row[index] = Table[HorBoundaries2[index][0]+5:HorBoundaries2[index][1]-5,0:width]
            HeightTemp = Row[index].shape[0]
            Row2[index] = [[] for index3 in range(len(TextLocs))]
            for index2 in range(len(Row2[index])):
                Row2[index][index2] = Row[index][0:HeightTemp, Boundaries2[index2][0]+5:Boundaries2[index2][1]-5]
                Markers.append([[HorBoundaries2[index][0],Boundaries2[index2][1]],[HorBoundaries2[index][1],Boundaries2[index2][0]]])
        VerFill =[]
        for index in range(len(Row2)):
            for index2 in range(len(Row2[index])):
                HeightTemp, WidthTemp = Row2[index][index2].shape
                FillList = []
                for index3 in range(WidthTemp):
                    index4 = 0
                    Filled = False
                    while (index4 < HeightTemp) and (not(Filled)):
                        if Row2[index][index2][index4, index3] != NegColor:
                            Filled = True
                        else:
                            index4 += 1
                    FillList.append(int(Filled))
                VerFill.append(FillList)
        for index in range(len(VerFill)):
            Sensitivity = int(math.floor(len(VerFill[index])/20))
            VerFill[index] = "".join(map(str, VerFill[index]))
            keys = []
            solution = []
            Original = copy.deepcopy(VerFill[index])
            for index2 in range(Sensitivity):
                keys.append("1" + index2*"0" + "1")
                solution.append("1"*(index2 + 2))
                VerFill[index] = VerFill[index].replace(keys[index2], solution[index2])
            VerFill[index]= list(map(int, list(VerFill[index])))
            VerFill2 = copy.deepcopy(VerFill)
            for key, group in itertools.groupby(VerFill[index]):
                VerFill2[index].append(key)
        Seperators = list(set(Rows + NegRows + [width, 0]))
        Seperators.sort()
        SumsPieces = [None for x in range(len(Seperators))]
        for index in range(len(Seperators)-1):
            if Seperators[index+1]- 5 - (Seperators[index]+5) >= 1:
                SumsPieces[index] = HorSums[Seperators[index]+5:Seperators[index+1]-5]
        TextRows = SumsPieces
        for index in range(len(VerSums)):
            if Distance(index, Seperators) <= 5:
                VerSums[index] = 0
    Sensitive = True
    s_5 = time.time()
    if Sensitive:
        VerSums2 = copy.deepcopy(VerSums)
        index = 0
        while VerSums[index] == 0:
            index = index + 1
        VerSums2 = VerSums2[index:]
        index = len(VerSums2)
        while VerSums2[index-1] == 0:
            index = index -1
        VerSums2 = VerSums2[:index]
        Zeros = []
        for index in range(0, len(VerSums)):
            if VerSums[index] == 0:
                Zeros.append(index)
    s_6 = time.time()


    Plot2 = VerSums2
    Plot1 = NeutVerSums2    
    if Demo:
        ax0.imshow(disp_img, vmin=image.min(), vmax=image.max(), cmap='gray')
        #for index in range(len(Markers)):
        #    ax0.Rectangle((Markers[index][0][0], Markers[index][0][1]), Markers[index][1][0]-Markers[0][0], Markers[index][1][1] - Markers[index][0][1])
        for index in range(0, len(Rows)):
             ax0.axhline(Rows[index], color="r")
        for index1 in range(0, len(Columns)):
            ax0.axvline(Columns[index1], color="r")
        for index in range(len(NegRows)):
            ax0.axhline(NegRows[index], color = "blue")
        for index in range(len(NegCols)):
            ax0.axvline(NegCols[index], color = "blue")
        #ax0.set_title('Table')
        #ax0.axis('off')
        #for index in Zeros:
        #    ax0.axvline(index, color = "yellow")
        #ax0.plot(Plot1, color = "black", label = "WHATEVER")
        #ax0.set_ylim(min(Plot1) - 0.1*min(Plot1), 2*max(Plot1))
        #for x in PotCol:
        #    ax0.plot(x, Plot1[x], "x", color = "red")
        #Factor = PotColPeakHeight/max(VerSums)
        #print(len(VerSums3))
        #ax0.axhline(Factor*max(Plot1), color = "red")
        #ax0.set_xlim(0, len(Plot1))


        ax1.plot(Plot2, color = "black", label = "Vertical Fill")
        ax1.set_ylim(0, 2*max(Plot2))
        ax1.axhline(CutOff, color = "red")
        #for index in Zeros:
        #    ax1.plot(index, VerSums[index], "x")


    if LogOutput and False:
        print("It's this bastard")
        print("This is a table with Width: " + str(len(Boundaries2)) + " and Height: " + str(len(HorBoundaries2)))

    Output = [file, len(Boundaries2), len(HorBoundaries2)]
    if PklOutput:
        pickle.dump(Output, open( "Output.p", "wb"))
    if Demo:
        plt.show()
        fig.tight_layout()
    Timer = True

    if Timer:
        print("\n")
        print("TIMER")
        print("0-1 \t" + str(s_1 - s_0))
        print("1-2 \t" + str(s_2 - s_1))
        print("1 - 1a\t" + str(s_1a - s_1))
        print("1a - 1b\t" + str(s_1b - s_1a))
        print("1b - 1c\t" + str(s_1c - s_1b))
        print("1c - 2\t" + str(s_2 - s_1c))
        print("2-3 \t" + str(s_3 - s_2))
        print("3-4 \t" + str(s_4 - s_3))
        print("4-5 \t" + str(s_5 - s_4))
        print("5-6 \t" + str(s_6 - s_5))

    return Output

#Main(file, LogOutput, Complex, False, PklOutput)
