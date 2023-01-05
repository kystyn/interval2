from csv import DictReader, reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy


numValues = 200

def readDataFromFile(file1='Канал 1_800nm_0.2.csv', file2='Канал 2_800nm_0.2.csv'):
    ch1U = []
    ch2U = []

    with open(file1, 'r') as f:
        csv_reader = DictReader(f, delimiter=';',)
        for row in csv_reader:
            ch1U.append(float(row['x'])) # actually row[0]

    with open(file2, 'r') as f:
        csv_reader = DictReader(f, delimiter=';')
        for row in csv_reader:
            ch2U.append(float(row['x'])) # actually row[0]

    ch1U = np.asarray(ch1U, dtype=float)
    ch2U = np.asarray(ch2U, dtype=float)

    return ch1U, ch2U


def plotData(X, Y, legends, colors, xylabels, title, show=True):
    fig, ax = plt.subplots()
    for x, y, legend, color, xylabel in zip(X, Y, legends, colors, xylabels):
        ax.plot(x, y, label=legend, color=color)
        ax.set_xlabel(xylabel[0])
        ax.set_ylabel(xylabel[1])
    ax.legend(prop={'size': 16})
    plt.title(title)
    if show:
        fig.show()
    return fig, ax


def buildLinearRegression(channelU):
    # cropping bad values for regression building
    start = 15
    end = 195

    arr2 = channelU[start:end]
    arr1 = np.arange(start, end).reshape((-1, 1))

    # coefficients calculation
    model = LinearRegression().fit(arr1, arr2)
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    a = model.coef_
    b =  model.intercept_

    regression = a * np.arange(0, numValues) + b
    return regression, a


def makeIntervals(channelU, regression, a):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(channelU),), dtype=float)
    tolerance.fill(5e-5)

    err = np.abs(channelU - regression)
    X_inter = np.empty(shape=(len(channelU), 2), dtype='float')
    X_inter_d = np.empty(shape=(len(channelU), 2), dtype='float')

    ind = np.arange(0, numValues)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:,0] = channelU - err - tolerance
    X_inter[:,1] = channelU + err + tolerance
    X_inter_d[:,0] = X_inter[:,0] - a * ind
    X_inter_d[:,1] = X_inter[:,1] - a * ind

    return X_inter_d


# R external estimation
def externalEstimateR(X1_inter_d, X2_inter_d):
    maxd1 = max(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1], X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])
    mind1 = min(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1], X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])

    for i in range(1, numValues):
        d1 = max(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1], X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        maxd1 = max(maxd1, d1)
        d1 = min(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1], X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        mind1 = min(mind1, d1)
    print("Rext1 = ", mind1)
    print("Rext2 = ", maxd1)
    return mind1, maxd1


def calcaulateJaccard(R, X1_inter_d, X2_inter_d):
    all_intervals = np.concatenate((X1_inter_d, R * X2_inter_d), axis=0)
    intersection = all_intervals[0]
    union = all_intervals[0]
    for i in range(1, len(all_intervals)):
        intersection = [max(intersection[0], all_intervals[i][0]), min(intersection[1], all_intervals[i][1])]
        union = [min(union[0], all_intervals[i][0]), max(union[1], all_intervals[i][1])]
    jc = (intersection[1] - intersection[0]) / (union[1] - union[0])
    return jc


def internalEstimateRJaccard(Rmin, Rmax, X1_inter_d, X2_inter_d):
    R_interval = np.linspace(Rmin - 0.2, Rmax + 0.2, 1000)
    Jaccars = []

    for R in R_interval:
        Jaccars.append(calcaulateJaccard(R, X1_inter_d, X2_inter_d))
    print('MAX Jaccard =', max(Jaccars))
    return R_interval, max(Jaccars), Jaccars, R_interval[np.argmax(Jaccars)]


def main():
    plt.rcParams['text.usetex'] = True
    U1, U2 = readDataFromFile()

    num = np.arange(0, numValues)

    plotData(
        (num, num), (U1, U2), ('First channel', 'Second channel'),
        ('red', 'green'), (('n', 'mV'), ('n', 'mV')), 'Raw data')

    reg1, a1 = buildLinearRegression(U1)
    reg2, a2 = buildLinearRegression(U2)

    X1 = makeIntervals(U1, reg1, a1)
    X2 = makeIntervals(U2, reg2, a2)

    fig, ax = plotData(
        (num, num), 
        (reg1, reg2),
        ('Regression 1', 'Regression 2'),
        ('blue', 'orange'),
        (('n', 'mV'), ('n', 'mV')),
        'Regression', show=False)

    x1err = X1[:, 1] - X1[:, 0]
    x2err = X2[:, 1] - X2[:, 0]
    ax.errorbar(num, U1, yerr=x1err, color='red', label='First channel')
    ax.errorbar(num, U2, yerr=x2err, color='green', label ='Second channel')
    fig.show()

    extRmin, extRmax = externalEstimateR(X1, X2)
    R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(extRmin, extRmax, X1, X2)
    fig, ax = plotData((R_int,), 
                        (Jaccard,), ('Jaccard index',), (None,),
                        (('$R_{21}$', 'Jaccard index'),), 'Jaccard index', show=False)

    xl = -20000
    xr = 20000
    yyr = 0
    prevj = -1
    for x, y in zip(R_int, Jaccard):
        if y * prevj < 0:
            if xl == -20000:
                xl = x
                yl = y
            else:
                xr = x
                yr = y
    ax.scatter(extRmin, calcaulateJaccard(extRmin, X1, X2), color='red', label=f'$R_{{min}}={extRmin:.3f}$')
    ax.scatter(extRmax, calcaulateJaccard(extRmax, X1, X2), color='red', label=f'$R_{{max}}={extRmax:.3f}$')
    ax.scatter(Ropt, JaccardOpt, color='red', label=f'$R_{{opt}}={Ropt:.3f}$')
    ax.plot([xl, xl], [-0.9, yl], 'r--')
    ax.plot([xr, xr], [-0.9, yr], 'r--')
    ax.plot([xl, xr], [0, 0], 'r--')
    ax.legend(prop={'size': 16})
    fig.show()

    fig, ax = plt.subplots()
    ax.errorbar(num, (X1[:,0] + X1[:,1]) / 2, yerr=x1err, color='red', label='Channel 1 without drift')
    ax.errorbar(num, Ropt * (X2[:,0] + X2[:,1]) / 2, yerr=x2err, color='green', label='Channel 2 without drift')
    ax.legend(prop={'size': 16})
    plt.title('Combined data')
    fig.show()
    print()


if __name__ == '__main__':
    main()
