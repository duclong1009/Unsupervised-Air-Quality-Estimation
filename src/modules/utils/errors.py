import numpy as np


def mean_squared_error(pred, gt):
    return ((pred - gt) ** 2).mean(axis=0)


def mean_absolute_error(pred, gt):
    return (abs(pred - gt)).mean(axis=0)


def mean_absolute_percentage_error(pred, gt):
    return abs((gt - pred) / gt).mean(axis=0)
