from csv import DictReader, reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from draw_data_status import\
     draw_data_status_template, get_residual, get_leverage,\
     add_point


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


def statuses(num, radius, X):
    yp_center = np.loadtxt(f'yp_center_{num}.mat')
    fig, ax = draw_data_status_template(title=f'Influences. Channel {num}. Radius ${radius} \cdot 10^{-4}$')
    
    for x, y in enumerate(X):
        l = get_leverage(x, y, yp_center)
        r = get_residual(x, y, yp_center)
        add_point((l, r), ax, 'bo' if x in np.arange(50, 150) else 'ko' if x in np.arange(0, 50) else 'wo',
        'left' if x == 0 else 'center' if x == 50 else 'right' if x == 150 else None)
    ax.legend()
    fig.show()


def main():
    plt.rcParams['text.usetex'] = True
    
    U1, U2 = readDataFromFile()

    X1 = makeIntervals(U1, 1, 0.0126997, 7.494845e-6, (0, 200))
    statuses(1, 1, X1)
    X1 = makeIntervals(U1, 3, 0.0126997, 7.494845e-6, (0, 200))
    statuses(1, 3, X1)
    X1 = makeIntervals(U1, 5, 0.0126997, 7.494845e-6, (0, 200))
    statuses(1, 5, X1)
    X1 = makeIntervals(U1, 6, 0.0126997, 7.494845e-6, (0, 200))
    statuses(1, 6, X1)
    X2 = makeIntervals(U2, 1, 0.014314, 8.1099e-06, (0, 200))
    statuses(2, 1, X2)
    X2 = makeIntervals(U2, 3, 0.014314, 8.1099e-06, (0, 200))
    statuses(2, 3, X2)
    X2 = makeIntervals(U2, 5, 0.014314, 8.1099e-06, (0, 200))
    statuses(2, 5, X2)
    X2 = makeIntervals(U2, 6, 0.014314, 8.1099e-06, (0, 200))
    statuses(2, 6, X2)
    
    print()

    


if __name__ == '__main__':
    main()
