"""
# My first app
Here's our first attempt at using data to create a table:
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import streamlit as st


def radon_transform(filename, detectors, angleDelta, offset, filter):
    image = io.imread(filename, as_gray=True)

    width = image.shape[0]
    height = image.shape[1]

    radius = max(width, height) / 2

    img_centre_x = width / 2;
    img_centre_y = height / 2;

    angles = np.arange(0, 360, angleDelta)
    offsets = np.linspace(-offset / 2, offset / 2, detectors)

    partial_output = []
    sinogram = []
    kernel = []

    # maska z prezentacji
    kernel_size = 21
    center = kernel_size // 2
    kernel = np.zeros(kernel_size)

    for k in range(kernel_size):
        if ((k - center) % 2 == 0):
            kernel[k] = 0
        else:
            kernel[k] = (-4 / np.pi ** 2) / (k - center) ** 2

    kernel[center] = 1

    # plt.plot(kernel)
    # plt.show()

    #print("Wyliczona maska")

    for angle in angles:
        start_x_coords, start_y_coords, end_x_coords, end_y_coords = [], [], [], []
        for k in range(detectors):
            start_x_coord = radius * np.cos(np.deg2rad(angle)) + img_centre_x
            start_y_coord = radius * np.sin(np.deg2rad(angle)) + img_centre_y

            end_x_coords.append(radius * np.cos(np.deg2rad(angle + offsets[k] + 180)) + img_centre_x)
            end_y_coords.append(radius * np.sin(np.deg2rad(angle + offsets[k] + 180)) + img_centre_y)

        lines_coords = []

        for line in range(detectors):
            lines_coords.append(
                Bresenham(int(start_x_coord), int(start_y_coord), int(end_x_coords[line]), int(end_y_coords[line])))

        lines_brightnesses = []

        for line in lines_coords:
            res = 0
            for i in range(len(line[0])):
                if (line[0][i] < width and line[1][i] < height):
                    res = res + image[line[0][i], line[1][i]]
            lines_brightnesses.append(res / len(line[0]))

        if (filter):
            lines_brightnesses = np.convolve(lines_brightnesses, kernel, mode='same')

        sinogram.append(lines_brightnesses)

        output = np.zeros((width, height))

        for i in range(len(lines_coords)):
            for j in range(len(lines_coords[i][0])):
                x = lines_coords[i][0][j]
                y = lines_coords[i][1][j]

                if (x < width and y < height):
                    output[x][y] += lines_brightnesses[i]

        output /= len(lines_coords)

        partial_output.append(output)
        print("█", end="")

    partial_avg_output = [np.zeros((width, height)) for i in range(len(angles))]

    for i in range(width):
        for j in range(height):
            suma = 0
            for iteration in range(0, len(angles)):
                suma += partial_output[iteration][i][j]
                partial_avg_output[iteration][i][j] = suma / (iteration + 1)
        print("█", end="")

    for i in range(len(partial_avg_output)):
        cv2.normalize(partial_avg_output[i], partial_avg_output[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        print("█", end="")

    return image, sinogram, partial_avg_output


def Bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) > abs(x1 - x0):
        swapped = True
        x0, y0, x1, y1 = y0, x0, y1, x1
    else:
        swapped = False
    m = (y1 - y0) / (x1 - x0) if x1 - x0 != 0 else 1
    q = y0 - m * x0
    if x0 < x1:
        xs = np.arange(np.floor(x0), np.ceil(x1) + 1, +1, dtype=int)
    else:
        xs = np.arange(np.ceil(x0), np.floor(x1) - 1, -1, dtype=int)
    ys = np.round(m * xs + q).astype(int)
    if swapped:
        xs, ys = ys, xs
    return xs, ys


# main
detectorsNumber = st.slider('ilość detektorów', 1, 500, 150)  # 👈 this is a widget
DeltaAflaSetpVaule = st.slider('Krok DeltaAlfa', 0.1, 40.0, 1.0)  # 👈 this is a widget
offsetValue = st.slider('Offset', 1, 300, 180)  # 👈 this is a widget
st.checkbox("Filtruj sinogram")

filePath = st.text_input('Ścieżka do pliku', "./Kolo.jpg")

st.button("Wykonaj")
######3
pixelArrayInput = st.image(filePath)

notusedinput, sinogram, average = radon_transform(filePath, detectorsNumber, DeltaAflaSetpVaule, offsetValue, "1")

#show_result(notusedinput, sinogram, average, x=5, step=1, value=len(average))

st.map(sinogram)
x = 180  # widgets.IntSlider(min=1, max=len(partial_avg_output)
#st.image(sinogram[:x])
# plt.imshow(, 'gray')

# plt.subplot(1, 3, 3)
# plt.imshow(partial_avg_output[x - 1], 'gray')
