import io
import os
from datetime import datetime
import zipfile

from operator import itemgetter
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pydicom
import matplotlib
import matplotlib.figure
import numpy as np
from skimage.feature import blob_log
from PIL import Image
import operator
from sys import platform
import zipfile

# Setup of initial output dictionary
graticule_upload_analysis = {}


# code from the utils.py file
def range_invert(arrayin):
    max_val = np.amax(arrayin)
    arrayout = arrayin / max_val
    min_val = np.amin(arrayout)
    arrayout = arrayout - min_val
    arrayout = 1 - arrayout  # inverting the range
    arrayout = arrayout * max_val

    return arrayout


def norm01(arrayin):
    min_val = np.amin(arrayin)  # normalizing
    arrayout = arrayin - min_val
    arrayout = arrayout / (np.amax(arrayout))  # normalizing the data

    return arrayout


# /code from the utils.py file


def running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):

        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1

        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def scalingAnalysis(ArrayDicom_o, dx, dy, title):  # determine scaling
    ArrayDicom = norm01(ArrayDicom_o)
    blobs_log = blob_log(
        ArrayDicom, min_sigma=1, max_sigma=5, num_sigma=20, threshold=0.15
    )  # run on windows, for some stupid reason exclude_border is not recognized in my distro at home

    point_det = []
    for blob in blobs_log:
        y, x, r = blob
        point_det.append((x, y, r))

    point_det = sorted(
        point_det, key=itemgetter(2), reverse=True
    )  # here we sort by the radius of the dot bigger dots are around the center and edges

    point_det = np.asarray(point_det)

    # now we need to select the most extreme left and right point
    # print(np.shape(ArrayDicom)[0] // 2)
    # print(abs(point_det[:6, 1] - np.shape(ArrayDicom)[0] // 2) < 10)
    point_sel = []
    for i in range(0, 6):
        if abs(point_det[i, 1] - np.shape(ArrayDicom)[0] // 2) < 10:
            point_sel.append(abs(point_det[i, :]))

    point_sel = np.asarray(point_sel)

    imax = np.argmax(point_sel[:, 0])
    imin = np.argmin(point_sel[:, 0])

    # print(point_sel[imax, :], point_sel[imin, :])
    distance = (
        np.sqrt(
            (point_sel[imax, 0] - point_sel[imin, 0])
            * (point_sel[imax, 0] - point_sel[imin, 0])
            * dx
            * dx
            + (point_sel[imax, 1] - point_sel[imin, 1])
            * (point_sel[imax, 1] - point_sel[imin, 1])
            * dy
            * dy
        )
        / 10.0
    )
    
    dz = 10*(distance - 10)

    # Plot the figure of scaling results and attach to QA+
    fig = UTILS.get_figure()
    ax = fig.gca()  # Get current axis

    ax.volume = ArrayDicom_o
    width = ArrayDicom_o.shape[1]
    height = ArrayDicom_o.shape[0]
    extent = (0, 0 + (width * dx), 0, 0 + (height * dy))
    img = ax.imshow(ArrayDicom_o, extent=extent, origin="lower")
    # img = ax.imshow(ArrayDicom_o)
    ax.set_xlabel("x distance [mm]")
    ax.set_ylabel("y distance [mm]")
    plot_title = "Graticule Scaling \n" + title[0] + " " + title[1]
    ax.set_title(plot_title, fontsize=16)

    ax.scatter(point_sel[imax, 0] * dx, point_sel[imax, 1] * dy)
    ax.scatter(point_sel[imin, 0] * dx, point_sel[imin, 1] * dy)

    # adding a horizontal arrow
    ax.annotate(
        s="",
        xy=(point_sel[imax, 0] * dx, point_sel[imax, 1] * dy),
        xytext=(point_sel[imin, 0] * dx, point_sel[imin, 1] * dy),
        arrowprops=dict(arrowstyle="<->", color="r"),
    )  # example on how to plot a double headed arrow

    ax.text(
        (width // 2.8) * dx,
        (height // 2 + 12) * dy,
        "Distance=" + str(round(distance, 4)) + " cm",
        rotation=0,
        fontsize=14,
        color="r",
    )
    ax.text(
        (width // 2.8) * dx,
        (height // 2 - 30) * dy,
        r"$\Delta$z=" + str(round(dz, 4)) + " cm",
        rotation=0,
        fontsize=14,
        color="r",
    )

    # Attach to QA+
    UTILS.write_file(plot_title + ".png", ax)

    return distance, fig, dz


def viewer(volume, dx, dy, center, title, textstr):
    fig2 = UTILS.get_figure()
    ax = fig2.gca()  # Get current axis

    ax.volume = volume
    width = volume.shape[1]
    height = volume.shape[0]
    extent = (0, 0 + (volume.shape[1] * dx), 0, 0 + (volume.shape[0] * dy))
    img = ax.imshow(volume, extent=extent)
    # img=ax.imshow(volume)
    ax.set_xlabel("x distance [mm]")
    ax.set_ylabel("y distance [mm]")
    # ax.set_xlabel('x pixel')
    # ax.set_ylabel('y pixel')
    ax.set_xlim(width * dx / 2 - 10, width * dx / 2 + 10)
    ax.set_ylim(height * dy / 2 - 10, height * dy / 2 + 10)

    # fig.suptitle('Image', fontsize=16)
    ax.set_title("Image Center Analysis", fontsize=16)
    ax.text((volume.shape[1] + 250) * dx, (volume.shape[0]) * dy, textstr)
    fig2.subplots_adjust(right=0.75)
    fig2.colorbar(img, ax=ax, orientation="vertical")
    # fig.canvas.mpl_connect('key_press_event', process_key_axial)

    return fig2, ax


def full_imageProcess(ArrayDicom_o, dx, dy, title):  # process a full image
    ArrayDicom = norm01(ArrayDicom_o)
    height = np.shape(ArrayDicom)[0]
    width = np.shape(ArrayDicom)[1]

    blobs_log = blob_log(
        ArrayDicom, min_sigma=1, max_sigma=5, num_sigma=20, threshold=0.15
    )  # run on windows, for some stupid reason exclude_border is not recognized in my distro at home

    center = []
    point_det = []
    for blob in blobs_log:
        y, x, r = blob
        point_det.append((x, y, r))

    point_det = sorted(
        point_det, key=itemgetter(2), reverse=True
    )  # here we sort by the radius of the dot bigger dots are around the center and edges

    # we need to find the centre dot as well as the larger dots on the sides of the image

    # for j in range(0, len(point_det)):
    #     x, y, r = point_det[j]
    #     center.append((int(round(x)), int(round(y))))

    # now that we have detected the centre we are going to increase the precision of the detected point
    im_centre = Image.fromarray(
        255
        * ArrayDicom[
            height // 2 - 20 : height // 2 + 20, width // 2 - 20 : width // 2 + 20
        ]
    )
    im_centre = im_centre.resize(
        (im_centre.width * 10, im_centre.height * 10), Image.LANCZOS
    )

    xdet_int, ydet_int = point_detect_singleImage(im_centre)
    xdet = int(width // 2 - 20) + xdet_int / 10
    ydet = int(height // 2 - 20) + ydet_int / 10

    center.append((xdet, ydet))

    textstr = ""

    # print("center=", center)
    fig, ax = viewer(range_invert(ArrayDicom_o), dx, dy, center, title, textstr)

    return fig, ax, center


def full_imageProcess_noGraph(ArrayDicom_o):  # process a full image
    ArrayDicom = norm01(ArrayDicom_o)
    height = np.shape(ArrayDicom)[0]
    width = np.shape(ArrayDicom)[1]

    blobs_log = blob_log(
        ArrayDicom, min_sigma=1, max_sigma=5, num_sigma=20, threshold=0.15
    )  # run on windows, for some stupid reason exclude_border is not recognized in my distro at home

    center = []
    point_det = []
    for blob in blobs_log:
        y, x, r = blob
        point_det.append((x, y, r))

    point_det = sorted(
        point_det, key=itemgetter(2), reverse=True
    )  # here we sort by the radius of the dot bigger dots are around the center and edges

    # we need to find the centre dot as well as the larger dots on the sides of the image

    # for j in range(0, len(point_det)):
    #     x, y, r = point_det[j]
    #     center.append((int(round(x)), int(round(y))))

    # now that we have detected the centre we are going to increase the precision of the detected point
    im_centre = Image.fromarray(
        255
        * ArrayDicom[
            height // 2 - 20 : height // 2 + 20, width // 2 - 20 : width // 2 + 20
        ]
    )
    im_centre = im_centre.resize(
        (im_centre.width * 10, im_centre.height * 10), Image.LANCZOS
    )

    xdet_int, ydet_int = point_detect_singleImage(im_centre)
    xdet = int(width // 2 - 20) + xdet_int / 10
    ydet = int(height // 2 - 20) + ydet_int / 10

    center.append((xdet, ydet))

    # fig, ax=viewer(u.range_invert(ArrayDicom_o), dx, dy, center, title, textstr)

    return center


def point_detect_singleImage(imcirclist):
    detCenterXRegion = []
    detCenterYRegion = []

    # print("Finding bibs in phantom...")
    grey_img = np.array(imcirclist, dtype=np.uint8)  # converting the image to grayscale
    blobs_log = blob_log(
        grey_img, min_sigma=15, max_sigma=50, num_sigma=10, threshold=0.05
    )

    centerXRegion = []
    centerYRegion = []
    centerRRegion = []
    grey_ampRegion = []
    for blob in blobs_log:
        y, x, r = blob
        # center = (int(x), int(y))
        centerXRegion.append(x)
        centerYRegion.append(y)
        centerRRegion.append(r)
        grey_ampRegion.append(grey_img[int(y), int(x)])

    xindx = int(centerXRegion[np.argmin(grey_ampRegion)])
    yindx = int(centerYRegion[np.argmin(grey_ampRegion)])
    # rindx = int(centerRRegion[np.argmin(grey_ampRegion)])

    detCenterXRegion = xindx
    detCenterYRegion = yindx

    return detCenterXRegion, detCenterYRegion


def read_dicom(directory):
    for subdir, dirs, files in os.walk(directory):
        dirs.clear()
        list_title = []
        list_gantry_angle = []
        list_collimator_angle = []
        list_figs = []
        center = []
        center_g0 = [(0, 0)]
        center_g90 = [(0, 0)]
        center_g180 = [(0, 0)]
        center_g270 = [(0, 0)]
        dx = 0
        dy = 0
        dzs = []
        distance = 0
        scaling_image = []
        k = 0

        for file in files:
            if os.path.splitext(directory + file)[1] == ".dcm":

                # Read the dicom file
                dataset = pydicom.dcmread(directory + file)

                # Write the raw images to a png file
                # UTILS.write_file(file + ".png", dataset)

                # Extract the gantry and collimator angles from the dicom files
                gantry_angle = dataset[0x300A, 0x011E].value
                collimator_angle = dataset[0x300A, 0x0120].value

                list_gantry_angle.append(gantry_angle)
                list_collimator_angle.append(collimator_angle)

                if round(gantry_angle) == 360:
                    gantry_angle = 0
                if round(collimator_angle) == 360:
                    collimator_angle = 0

                title = (
                    "g" + str(round(gantry_angle)),
                    "c" + str(round(collimator_angle)),
                )

                if k == 0:
                    title = (
                        "g" + str(round(gantry_angle)),
                        "c" + str(round(collimator_angle)),
                    )
                    list_title.append(title)
                    ArrayDicom = dataset.pixel_array
                    height = np.shape(ArrayDicom)[0]
                    width = np.shape(ArrayDicom)[1]
                    SID = dataset.RTImageSID

                    # Pixel Spacing to convert pixels to real world dimensions
                    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
                    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
                    _, fig_scaling, dz = scalingAnalysis(ArrayDicom, dx, dy, title)
                    scaling_image.append(fig_scaling)
                    

                else:
                    list_title.append(title)
                    tmp_array = dataset.pixel_array
                    tmp_array = norm01(tmp_array)
                    ArrayDicom = np.dstack((ArrayDicom, tmp_array))
                    _, fig_scaling, dz = scalingAnalysis(tmp_array, dx, dy, title)
                    scaling_image.append(fig_scaling)
                    
                dzs.append(dz)
                k = k + 1

                # Delete the file afterwords
                os.remove(directory + file)

    # After we colect all the images we only select g0c90 and g0c270 to calculate the center at g0
    k = 0
    l = 0
    m = 0
    n = 0

    for i, _ in enumerate(list_title):
        if list_title[i][0] == "g0" and k == 0:
            k = k + 1
            # height = np.shape(ArrayDicom[:, :, i])[0]
            # width = np.shape(ArrayDicom[:, :, i])[1]
            fig_g0c90, ax_g0c90, center_g0c90 = full_imageProcess(
                ArrayDicom[:, :, i], dx, dy, list_title[i]
            )
            center_g0[0] = (
                center_g0[0][0] + center_g0c90[0][0],
                center_g0[0][1] + center_g0c90[0][1],
            )

            list_figs.append(fig_g0c90)  # we plot always the image at g0c90

        elif list_title[i][0] == "g0" and k != 0:
            k = k + 1
            center_g0c270 = full_imageProcess_noGraph(ArrayDicom[:, :, i])
            center_g0[0] = (
                center_g0[0][0] + center_g0c270[0][0],
                center_g0[0][1] + center_g0c270[0][1],
            )

    for i, _ in enumerate(list_title):
        if list_title[i][0] == "g90":
            l = l + 1
            center_g90c = full_imageProcess_noGraph(ArrayDicom[:, :, i])
            center_g90[0] = (
                center_g90[0][0] + center_g90c[0][0],
                center_g90[0][1] + center_g90c[0][1],
            )

        if list_title[i][0] == "g180":
            m = m + 1
            center_g180c = full_imageProcess_noGraph(ArrayDicom[:, :, i])
            center_g180[0] = (
                center_g180[0][0] + center_g180c[0][0],
                center_g180[0][1] + center_g180c[0][1],
            )

        if list_title[i][0] == "g270":
            n = n + 1
            center_g270c = full_imageProcess_noGraph(ArrayDicom[:, :, i])
            center_g270[0] = (
                center_g270[0][0] + center_g270c[0][0],
                center_g270[0][1] + center_g270c[0][1],
            )

    center_g0[0] = (center_g0[0][0] / k, center_g0[0][1] / k)
    center_g90[0] = (center_g90[0][0] / l, center_g90[0][1] / l)
    center_g180[0] = (center_g180[0][0] / m, center_g180[0][1] / m)
    center_g270[0] = (center_g270[0][0] / n, center_g270[0][1] / n)

    center.append(center_g0[0])
    center.append(center_g90[0])
    center.append(center_g180[0])
    center.append(center_g270[0])

    x_g0, y_g0 = center_g0[0]
    x_g90, y_g90 = center_g90[0]
    x_g180, y_g180 = center_g180[0]
    x_g270, y_g270 = center_g270[0]

    max_deltax = 0
    max_deltay = 0
    max_deltaz = 0
    for i in range(0, len(center)):
        for j in range(i + 1, len(center)):
            deltax = abs(center[i][0] - center[j][0])
            deltay = abs(center[i][1] - center[j][1])
            if deltax > max_deltax:
                max_deltax = deltax
            if deltay > max_deltay:
                max_deltay = deltay
    
    #Calculate the max dz
    abs_dzs = list(map(abs,dzs))
    max_deltaz = dzs[abs_dzs.index(max(abs_dzs))]
    
    # Save the calculated answers to the output dictionary for output into other QA+ composites
    graticule_upload_analysis["Max Delta X"] = max_deltax * dx
    graticule_upload_analysis["Max Delta Y"] = max_deltay * dy
    graticule_upload_analysis["Max Delta Z"] = max_deltaz

    ax_g0c90.scatter(
        x_g0 * dx, (ArrayDicom[:, :, i].shape[0] - y_g0) * dy, label="g=0"
    )  # perfect!

    ax_g0c90.scatter(
        x_g90 * dx, (ArrayDicom[:, :, i].shape[0] - y_g90) * dy, label="g=90"
    )  # perfect!
    ax_g0c90.scatter(
        x_g180 * dx, (ArrayDicom[:, :, i].shape[0] - y_g180) * dy, label="g=180"
    )  # perfect!
    ax_g0c90.scatter(
        x_g270 * dx, (ArrayDicom[:, :, i].shape[0] - y_g270) * dy, label="g=270"
    )  # perfect!

    # print(list_title[i], "center_g0c90=", center_g0c90, "center=", center, dist)
    ax_g0c90.legend(bbox_to_anchor=(1.25, 1), loc=2, borderaxespad=0.0)

    # adding a horizontal arrow
    # ax.annotate(
    #     s="",
    #     xy=(point_sel[imax, 0] * dx, point_sel[imax, 1] * dy),
    #     xytext=(point_sel[imin, 0] * dx, point_sel[imin, 1] * dy),
    #     arrowprops=dict(arrowstyle="<->", color="r"),
    # )  # example on how to plot a double headed arrow
    ax_g0c90.text(
        (width // 2.15) * dx,
        (height // 2.13) * dy,
        "Maximum delta x =" + str(round(max_deltax * dx, 4)) + " mm",
        rotation=0,
        fontsize=14,
        color="r",
    )
    ax_g0c90.text(
        (width // 2.15) * dx,
        (height // 2.16) * dy,
        "Maximum delta y =" + str(round(max_deltay * dy, 4)) + " mm",
        rotation=0,
        fontsize=14,
        color="r",
    )
    ax_g0c90.text(
        (width // 2.15) * dx,
        (height // 2.19) * dy,
        "Maximum delta z =" + str(round(max_deltaz, 4)) + " cm",
        rotation=0,
        fontsize=14,
        color="r",
    )

    # Attach ax_g0c90 to QA+
    UTILS.write_file("isocenter_results.png", ax_g0c90)


# Load the zip file and unpack into a temporary folder
zip_file = BIN_FILE  # zip file upload
zfiles = zipfile.ZipFile(zip_file)
zfiles.extractall(path="tmp")

read_dicom("tmp/")

# close the file and delete
zfiles.close()
del zfiles
