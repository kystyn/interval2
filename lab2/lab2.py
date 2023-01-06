from csv import DictReader, reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog


numValues = 200

def readDataFromFile(file1='data1.csv', file2='data2.csv'):
    ch1U = []
    ch2U = []

    with open(file1, 'r') as f:
        csv_reader = reader(f, delimiter=' ',)
        for row in csv_reader:
            if len(row):
                ch1U.append(float(row[0])) # actually row[0]

    with open(file2, 'r') as f:
        csv_reader = reader(f, delimiter=' ')
        for row in csv_reader:
            if len(row):
                ch2U.append(float(row[0])) # actually row[0]

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


def makeIntervals(channelU, weights, beta0, beta1, rng):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(channelU),), dtype=float)
    tolerance.fill(1e-4)

    tolerance = tolerance * weights
    err = np.abs(channelU - beta0)
    ind = np.arange(*rng)
    
    X = np.empty(shape=(len(channelU), 2), dtype='float')
    X[:, 0] = channelU - err - tolerance - beta1 * ind
    X[:, 1] = channelU + err + tolerance - beta1 * ind

    return X
    

def calcaulateJaccard(R, X1, X2):
    all_intervals = np.concatenate((X1, R * X2), axis=0)
    intersection = all_intervals[0]
    union = all_intervals[0]
    for i in range(1, len(all_intervals)):
        intersection = [max(intersection[0], all_intervals[i][0]), min(intersection[1], all_intervals[i][1])]
        union = [min(union[0], all_intervals[i][0]), max(union[1], all_intervals[i][1])]
    jc = (intersection[1] - intersection[0]) / (union[1] - union[0])
    return jc


def internalEstimateRJaccard(Rmin, Rmax, X1_inter_d, X2_inter_d):
    R_interval = np.linspace(Rmin, Rmax, 1000)
    Jaccars = []

    for R in R_interval:
        Jaccars.append(calcaulateJaccard(R, X1_inter_d, X2_inter_d))
    print('MAX Jaccard =', max(Jaccars))
    return R_interval, max(Jaccars), Jaccars, R_interval[np.argmax(Jaccars)]


def ir_outer(U):
    c = np.asarray([1, 0])
    A = np.concatenate(
        (
            np.asarray([[1, i] for i in range(1, 201)]),
            np.asarray([[-1, -i] for i in range(1, 201)])
        ), axis=0
    )
    
    b = np.asarray([u + 0.00065 for u in U] + [-u + 0.00065 for u in U])
    b = b.reshape((400, 1))
    bounds = [(None, None) for _ in range(2)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
    return res


def makePartialInterval():
    U1, U2 = readDataFromFile()

    #ir_outer(U1)

    X1_1 = makeIntervals(U1[0:50],    6.54538, 0.012174, 2.0065e-5, (0, 50) ) ##
    X1_2 = makeIntervals(U1[50:150],  1,       0.012699, 7.4948e-6, (50, 150))
    X1_3 = makeIntervals(U1[150:200], 1.6666,  0.011491, 1.4171e-5, (150, 200))

    X1 = np.concatenate((X1_1, X1_2, X1_3))

    
    X2_1 = makeIntervals(U2[0:50],    2.31319, 0.01420, 1.3681e-5, (0, 50)) 
    X2_2 = makeIntervals(U2[50:150],  1,       0.01431, 8.1099e-6, (50, 150)) 
    X2_3 = makeIntervals(U2[150:200], 1,       0.01318, 1.4310e-5, (150, 200)) 

    X2 = np.concatenate((X2_1, X2_2, X2_3))

    return X1, X2, (X1_1, X1_2, X1_3), (X2_1, X2_2, X2_3)


def main():
    plt.rcParams['text.usetex'] = True
    
    X1, X2, _, _ = makePartialInterval()

    num = np.arange(0, numValues)
    x1err = X1[:, 1] - X1[:, 0]
    x2err = X2[:, 1] - X2[:, 0]
    U1, U2 = readDataFromFile()
    fig, ax = plt.subplots()
    ax.errorbar(num, U1, yerr=x1err, color='red', label='First channel')
    ax.set_xlabel('N')
    ax.set_ylabel('U, mV')
    ax.legend(prop={'size': 16})
    plt.title('Model 1')
    fig.show()

    fig, ax = plt.subplots()
    ax.errorbar(num, U2, yerr=x2err, color='green', label ='Second channel')
    ax.set_xlabel('N')
    ax.set_ylabel('U, mV')
    ax.legend(prop={'size': 16})
    plt.title('Model 2')
    fig.show()


    R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(0.7, 1.0, X1, X2)
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
    ax.scatter(0.7, calcaulateJaccard(0.7, X1, X2), color='red', label=f'$R_{{min}}={0.7:.3f}$')
    ax.scatter(1.0, calcaulateJaccard(1.0, X1, X2), color='red', label=f'$R_{{max}}={1.0:.3f}$')
    ax.scatter(Ropt, JaccardOpt, color='red', label=f'$R_{{opt}}={Ropt:.3f}$')
    ax.plot([xl, xl], [-0.9, yl], 'r--')
    ax.plot([xr, xr], [-0.9, yr], 'r--')
    ax.plot([xl, xr], [0, 0], 'r--')
    ax.legend(prop={'size': 16})
    fig.show()
    print()


if __name__ == '__main__':
    main()
