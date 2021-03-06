"""
TestUtilityModule.py script tests all cases for PhiRelevance module
imports PhiRelevance module --> Calls functions of PhiUtils script of PhiRelevance module and plots graphs between target continous variable and relevance.
"""

import warnings
warnings.filterwarnings("ignore")
from PhiRelevance.PhiUtils import phiControl,phi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plotRelevance(y,relevance):
    plt.plot(y,relevance,'ro')
    plt.xlabel('Class Labels')
    plt.ylabel('relevance')
    plt.show()


def callUtilityFunctions(data,method,extrType, controlPts, coef):
    if(method=="extremes"):
        controlPts, npts = phiControl(data,method,extrType, controlPts, coef)
    else:
        controlPts, npts = phiControl(data,method,extrType, controlPts, coef)
    print("------------------------------------------")
    print("Control Points", controlPts)
    print("npts:", npts)
    print("------------------------------------------")

    if(controlPts == -1 and npts == -1):
        print("Invalid Parameters")
        print("------------------------------------------")
        return [] , [] , []

    yPhi, ydPhi, yddPhi = phi(data, controlPts, npts, 'extremes')
    return yPhi, ydPhi, yddPhi


if __name__ == '__main__':

    data = pd.read_csv("./datasets/bike-sharing/hour.csv")
    y = data["cnt"]


    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "extremes", 'high',[], 0.05)

    plotRelevance(y, yPhi)


    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "extremes", 'low', [], 0.05)

    plotRelevance(y, yPhi)

    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "extremes", 'both',[], 0.05)
    plotRelevance(y, yPhi)

    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "range", "" ,[[1,1,0],[2,1,0],[3,1,1]],-1)
    plotRelevance(y, yPhi)

    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "range", "" ,[[1,1],[2,1],[3,1]],-1)
    plotRelevance(y, yPhi)

    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "range", "" ,[[1],[2],[3]],-1)

    yPhi, ydPhi, yddPhi = callUtilityFunctions(y, "range", "" ,[],-1)

