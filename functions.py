
import numpy as np
import math
from skimage import img_as_float32


def add_all_noises(input_image: np.ndarray, strip_dens: float = 0.3, random_noise_intensity=0.1,
                   gaus_factor=0.1) -> np.ndarray:
    """
    add_all_noises adds 3 types of noises to the input image: strip bias noise, random noise(S&P)
    and a gaussian bias noise. All of this noises can be manipulated by the input's parameters.

    Parameters:
      *input_image : 2D array
      *strip_dens : float(optional), defaulted to 0.3
      *random_noise_intensity : float(optional), defaulted to 0.1
      *gaus_factor : float(optional), defaulted to 0.1

    Returns:
      Noised image : 2D array
      Noise only : 2D array
    """
    y, x = np.shape(input_image)  # y = number of rows, x = number of columns

    # strip bias noise - linear statistic model(able to modulate a second order
    # polinomial model)
    random_base = np.random.normal(0, 0.2, size=x)  # produce a random array for manipulating the stripes later
    norm_image = np.divide(input_image, 255)  # normalize the input image to be (0-1) for each pixel
    for colm in range(x):  # iterate on all the columns
        if (random_base[colm] > strip_dens):  # 0.0668 to succeed(gaussian(std=0.2)>0.3=0.0668)
            random_mean = random_base[
                colm].mean()  # first order polinomial model, each chosen colmn is multiplied by (1.3-2.3)
            norm_image[:, colm] = (1 + random_mean) * norm_image[:,
                                                      colm]  # first order polinomial model, each chosen column is multiplied by (1.3-2.3)

    FPN = norm_image - np.divide(input_image, 255)
    # #random noise(S&P)
    noise = np.random.normal(0, random_noise_intensity,
                             size=np.shape(input_image))  # produce a random values 2D array with input impage's shape
    # can be manipulated by random_noise_intensity input
    output = np.add(norm_image, noise)  # adding the random noise to the input image

    # gaussian bias noise
    ground = np.zeros(shape=np.shape(input_image))  # produce an empty 2D array with input impage's shape
    gaus_factor = np.random.normal(0, 1)
    max_val = math.pow((int)(gaus_factor * (x / 2)), 2) + math.pow((int)(gaus_factor * (y / 2)),
                                                                   2)  # calculate the maximum possible value for later normalization
    for colm in range(x):
        for row in range(y):
            n_factor = math.pow((int)(gaus_factor * (x / 2 - colm)), 2) + math.pow((int)(gaus_factor * (y / 2 - row)),
                                                                                   2)  # calculate each pixel n_factor via its location and the random gaus_factor
            if max_val == 0:
                max_val = 0.001
            pixel = math.pow(((max_val - n_factor) / max_val), 6)  # normlize the outcome and power it up
            ground[row][colm] = pixel  # locate the pizel back to the empty 2D array

    output = np.add(output, ground)  # adding gaussian 2D array to the output image
    norm = np.max(output)
    output = output/norm
    output = np.clip(output, 0, 1)
    output = img_as_float32(output)
    all_noises = 2.5*FPN + noise/2 + ground/3
    all_noises = all_noises/2
    all_noises = np.clip(all_noises, 0, 1)
    all_noises = img_as_float32(all_noises)
    return output, all_noises

def add_no_noises(input_image: np.ndarray, strip_dens: float = 0.3, random_noise_intensity=0.1,
                   gaus_factor=0.1) -> np.ndarray:
    y, x = np.shape(input_image)
    norm_image = np.divide(input_image, 255)
    no_noise = np.random.normal(0, 0, size=np.shape(input_image))
    output = np.add(norm_image, no_noise)
    norm = np.max(output)
    output = output/norm
    output = np.clip(output, 0, 1)
    output = img_as_float32(output)
    return output