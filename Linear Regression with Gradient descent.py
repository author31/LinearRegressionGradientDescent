import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x= [1, 2, 4, 3, 5]
y= [1, 3, 3, 2, 5]

def GD(mCurrent,bCurrent,learningRates):
    error = 0
    for i,z in zip(x,y):
        yhat = mCurrent*i +bCurrent
        error += (yhat- z)**2
        mCurrent += (error*i)*learningRates
        bCurrent += (error)*learningRates
    return mCurrent,bCurrent

GD(0.8,0.4,0.005)


def newLR(x,y,learningRates):
    #calculate coefficients m and b
    n = 0
    d = 0
    xbar = sum(x)/len(x)
    ybar = sum(y)/len(x)
    for i,z in zip(x,y):
        n += (i-xbar)*(z-ybar)
        d += (i-xbar)**2
        m = n/d
        b = ybar -m*xbar
        b = round(b,2)
    #predict the new Y values
        yhat = m*i +b
    #calculate the r_squared before GD    
        squareYhat = np.square(z-yhat)
        squareYbar = np.square(z-ybar)
        rSquared = 1-(squareYhat/squareYbar)
    #using GD to optimize coef(s)
        mCurrent = GD(m,b,learningRates)[0]
        bCurrent = GD(m,b,learningRates)[1]
    #predcit new Y values with optimized coef(s)
        newYhat = mCurrent*i+bCurrent
    #calculate the r_squared after GD    
        gdYhat = np.square(z-newYhat)
        gdYbar = np.square(z-ybar)
        gdSquared = 1-(gdYhat/gdYbar)
    return rSquared, gdSquared

newLR(x,y,0.005)