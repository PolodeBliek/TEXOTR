import numpy as np
import matplotlib.pyplot as plt
import random
import copy

from scipy.ndimage import gaussian_filter
from skimage import data, io
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from skimage.exposure import histogram
import more_itertools as mit
import math


def avg(l):
    return (sum(l)/len(l))

def Distance(Number, List):
    ListTemp = copy.deepcopy(List)
    ListTemp = [abs(x-Number) for x in ListTemp]
    return (min(ListTemp))


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# Convert to float: Important for subtraction later which won't work with uint8
image = img_as_float(io.imread("ImExamples/example9.png"))

(height,width, notimportant) = image.shape

dilated = img_as_float(rgb2gray(image))
hist, hist_centers = histogram(dilated)


h = 0.4
Complex = True



fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))

slices = []
verslices = []
clearslices = []
clearverslices = []
completelist = []

for index in range(0, width):
    verslices.append([])
for index in range(0, height):
    slices.append(dilated[index].tolist())
    completelist = completelist + dilated[index].tolist()
    for index2 in range(0, width):
        verslices[index2].append(dilated[index, index2])

for index in range(height):
    if max(slices[index])-min(slices[index])<0.1:
        clearslices.append(index)
for index in range(width):
    if max(verslices[index])-min(verslices[index])<0.1:
        clearverslices.append(index)


sums = [1.0 - (1/height)*sum(x) for x in zip(*slices)]
versums = [1.0 - (1/width)*sum(x) for x in zip(*verslices)]

potcolpeakheight = max(sums)*0.5 if max(sums)*0.5 > 0.10 else 0.10
potrowpeakheight = max(versums)*0.5 if max(versums)*0.5 > 0.10 else 0.10

colunifac = []
rowunifac = []

potcolpeaks, _ = find_peaks(sums, height = potcolpeakheight)
potrowpeaks, _ = find_peaks(versums, height = potrowpeakheight)

potcolpeaks2 = []
potrowpeaks2 = []

for index in range(len(potcolpeaks)):
    potcolpeakspeaks, _ = find_peaks(verslices[potcolpeaks[index]], height = min(verslices[potcolpeaks[index]]) + (max(verslices[potcolpeaks[index]])-min(verslices[potcolpeaks[index]]))*0.5)
    potcolpeakspeaks = list(set(potcolpeakspeaks)-set(clearslices)-set(potrowpeaks))
    if len(potcolpeakspeaks) == 0:
        potcolpeaks2.append(potcolpeaks[index])
    else:
        pass

for index in range(len(potrowpeaks)):
    potrowpeakspeaks, _ = find_peaks(slices[potrowpeaks[index]], height = min(slices[potrowpeaks[index]]) + (max(slices[potrowpeaks[index]])-min(slices[potrowpeaks[index]]))*0.5)
    potrowpeakspeaks = list(set(potrowpeakspeaks)-set(clearverslices)-set(potcolpeaks))
    if len(potrowpeakspeaks) == 0:
        potrowpeaks2.append(potrowpeaks[index])
    else:
        pass


colpeaks = potcolpeaks2
rowpeaks = potrowpeaks2
nocolpeaks = list(set(potcolpeaks)-set(colpeaks))
norowpeaks = list(set(potrowpeaks)-set(rowpeaks))

clearslices2 = []
clearverslices2 = []

if len(rowpeaks) != 0:
    for index in range(len(clearslices)):
        dist = [abs(i-clearslices[index]) for i in rowpeaks]
        if min(dist) <= 10:
            pass
        else:
            clearslices2.append(clearslices[index])

if len(colpeaks) != 0:
    for index in range(len(clearverslices)):
        dist = [abs(i-clearverslices[index]) for i in colpeaks]
        if min(dist) <= 10:
            pass
        else:
            clearverslices2.append(clearverslices[index])

whiteslices = [list(group) for group in mit.consecutive_groups(clearslices2)]
whiteverslices = [list(group) for group in mit.consecutive_groups(clearverslices2)]
whiteslices = [int(sum(group)/len(group)) for group in whiteslices]
whiteverslices = [int(sum(group)/len(group)) for group in whiteverslices]

if Complex:
    #Gaps are the places where it believes the text is
    VerSeperators = list(set(colpeaks + whiteverslices + [width, 0]))
    VerSeperators.sort()
    VerSumsPieces = [None for x in range(len(VerSeperators))]
    for index in range(len(VerSeperators)-1):
        if VerSeperators[index+1]- 5 - (VerSeperators[index]+5) >= 1:
            VerSumsPieces[index] = versums[VerSeperators[index]+5:VerSeperators[index+1]-5]
    TextColumns = []
    for element in VerSumsPieces:
        if type(element)==list and len(element) != 0:
            TextColumns.append(element)
    print(len(TextColumns))
    for index in range(len(sums)):
        if Distance(index, VerSeperators) <= 5:
            sums[index] = 0
    SymmetryFactors = []
    for index in range(len(TextColumns)):
        for index2 in range(len(TextColumns[index])):
            pass

    Seperators = list(set(rowpeaks + whiteslices + [width, 0]))
    Seperators.sort()
    SumsPieces = [None for x in range(len(Seperators))]
    for index in range(len(Seperators)-1):
        if Seperators[index+1]- 5 - (Seperators[index]+5) >= 1:
            SumsPieces[index] = sums[Seperators[index]+5:Seperators[index+1]-5]
    TextRows = SumsPieces
    for index in range(len(versums)):
        if Distance(index, Seperators) <= 5:
            versums[index] = 0




ax0.plot(sums, color = "black",  label="Som")
for index in range(len(colpeaks)):
    ax0.plot(colpeaks[index], sums[colpeaks[index]], "x", color="r")
for index1 in range(len(nocolpeaks)):
    ax0.plot(nocolpeaks[index1], sums[nocolpeaks[index1]], "x", color = "orange")
ax0.set_ylim(0, 2)
ax0.set_title('Som')
ax0.set_xticks([])
ax0.legend()

ax1.plot(versums, color = "black",  label="Verticale Som")
for index in range(len(rowpeaks)):
    ax1.plot(rowpeaks[index], versums[rowpeaks[index]], "x", color="r")
for index1 in range(0, len(norowpeaks)):
    ax1.plot(norowpeaks[index1], versums[norowpeaks[index1]], "x", color = "orange")
ax1.set_ylim(0, 2)
ax1.set_title('Som')
ax1.set_xticks([])
ax1.legend()

ax2.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax2.set_title('Table')
for index in range(0, len(rowpeaks)):
     ax2.axhline(rowpeaks[index], color="r")
for index1 in range(0, len(colpeaks)):
    ax2.axvline(colpeaks[index1], color="r")
for index in range(len(whiteslices)):
    ax2.axhline(whiteslices[index], color = "blue")
for index in range(len(whiteverslices)):
    ax2.axvline(whiteverslices[index], color = "blue")
ax2.axis('off')

print("This is a table with Width: " + str(len(colpeaks)-1) + " and Height: " + str(len(rowpeaks)-1))

fig.tight_layout()
plt.show()
