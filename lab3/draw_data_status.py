import matplotlib.pyplot as plt
import numpy as np


def get_rad(interval):
    return (max(interval) - min(interval)) / 2


def get_mid(interval):
    return (interval[0] + interval[1]) / 2


def draw_data_status_template(x_lims=(0, 2), title='Influences'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('yellow')
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(-(x_lims[1] + 1), x_lims[1] + 1)
    # draw green triangle zone
    x1, y1 = [0, 1], [-1, 0]
    x2, y2 = [0, 1], [1, 0]
    ax.plot(x1, y1, 'k', x2, y2, 'k')
    ax.fill_between(x1, y1, y2, facecolor='green')

    # draw others zones
    x1, y1 = [0, x_lims[1]], [-1, -(x_lims[1] + 1)]
    x2, y2 = [0, x_lims[1]], [1, x_lims[1] + 1]

    ax.plot(x1, y1, 'k', x2, y2, 'k')
    x = np.arange(0.0, x_lims[1], 0.01)
    y1 = x + 1
    y2 = [x_lims[1] + 1] * len(x)
    ax.fill_between(x, y1, y2, facecolor='red')
    y2 = [-(x_lims[1] + 1)] * len(x)
    ax.fill_between(x, -y1, y2, facecolor='red')

    x1, y1 = [1, 1], [-(x_lims[1] + 1), x_lims[1] + 1]
    ax.plot(x1, y1, 'k--')
    ax.set_xlabel('l(x, y)')
    ax.set_ylabel('r(x, y)')
    ax.set_title(title)
    return fig, ax


def add_point(point, ax, color, label=None):
    ax.plot(point[0], point[1], color,label=label)


def get_intersections(interval_list):
    res = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res = [max(min(res), min(interval_list[i])), min(max(res), max(interval_list[i]))]
    return res


def get_intersections_wrong_int(interval_list):
    res = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res = [max(res[0], interval_list[i][0]), min(res[1], interval_list[i][1])]
    return res


def get_influences(interval_list, intersection_=None):
    if intersection_ is not None:
        intersection = intersection_
    else:
        intersection = get_intersections(interval_list)
    inter_rad = get_rad(intersection)
    inter_mid = get_mid(intersection)
    influences = []
    for interval in interval_list:
        l = inter_rad / get_rad(interval)
        r = (get_mid(interval) - inter_mid) / get_rad(interval)
        influences.append([l, r])
    return influences, intersection


def get_residuals(interval_d, edge_points, drift_params_3):
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list__ = []
        for num, drift_param in enumerate(drift_params_3[list_num]):
            if num == 0:
                new_list_ = list_[:edge_points[list_num][0]]
                start = 0
            elif num == 1:
                new_list_ = list_[edge_points[list_num][0]:edge_points[list_num][1]]
                start = edge_points[list_num][0]
            else:
                new_list_ = list_[edge_points[list_num][1]:]
                start = edge_points[list_num][1]
            for num_, interval in enumerate(new_list_, start=start):
                new_list__.append([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]])
        new_list.append(new_list__)
    return new_list


def get_residuals_1(interval_d, drift_params):
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list__ = []
        for num, drift_param in enumerate(drift_params[list_num]):
            for num_, interval in enumerate(list_, start=0):
                new_list__.append([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]])
        new_list.append(new_list__)
    return new_list


def get_residual(x, y, forecast):
    return (get_mid(y) - get_mid(forecast[x])) / get_rad(y)


def get_leverage(x, y, forecast):
    return get_rad(forecast[x]) / get_rad(y)
