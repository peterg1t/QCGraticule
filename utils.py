import numpy as np

def range_invert(arrayin):
    max_val = np.amax(arrayin)
    arrayout = arrayin / max_val
    min_val = np.amin(arrayout)
    arrayout = arrayout - min_val
    arrayout = (1 - arrayout)  # inverting the range
    arrayout = arrayout * max_val

    return arrayout


def norm01(arrayin):
    min_val = np.amin(arrayin)  # normalizing
    arrayout = arrayin - min_val
    arrayout = arrayout / (np.amax(arrayout))  # normalizing the data

    return arrayout

