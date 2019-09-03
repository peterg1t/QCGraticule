



###########################################################################################
#
#   Script name: qc-epidmtf
#
#   Description: Tool for calculating epid modlation transfer function (MTF) using a QCMV phantom (Standard Imaging).
#
#   Example usage: python qc-epidmtf "/file/"
#
#   Author: Pedro Martinez
#   pedro.enrique.83@gmail.com
#   5877000722
#   Date:2019-04-09
#
###########################################################################################

import os
import sys

# sys.path.append('C:\Program Files\GDCM 2.8\lib')
import pydicom
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import argparse
import cv2
from skimage.feature import blob_log
from math import *
from operator import itemgetter
from scipy.integrate import newton_cotes
import utils as u


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


# axial visualization and scrolling
def viewer(volume, dx, dy,center,title,textstr):
    # remove_keymap_conflicts({'j', 'k'})
    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()
    ax.volume = volume
    extent = (0, 0 + (volume.shape[1] * dx),
              0, 0 + (volume.shape[0] * dy))
    img=ax.imshow(volume, extent=extent)
    # img=ax.imshow(volume)
    ax.set_xlabel('x distance [mm]')
    ax.set_ylabel('y distance [mm]')
    # ax.set_xlabel('x pixel')
    # ax.set_ylabel('y pixel')

    # fig.suptitle('Image', fontsize=16)
    ax.set_title(title, fontsize=16)

    # for i in range(0,len(poly)): #maybe at a later stage we will add polygons drawings
    #     ax.add_patch(poly[i])
    ax.text((volume.shape[1]+250)*dx,(volume.shape[0])*dy,textstr)
    fig.subplots_adjust(right=0.75)
    fig.colorbar(img, ax=ax, orientation='vertical')
    # fig.canvas.mpl_connect('key_press_event', process_key_axial)
    for x,y in center:
        # ax.scatter(x,y)
        ax.scatter(x*dx+dx/2,(volume.shape[0]-y)*dy-dy/2) #adding dx/2 and subtracting dy/2 correctly puts the point in the center of the pixel when using extents and not in the edge.

    return fig







def shape_detect(c):
    shape='unidentified'
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c) #number of vertices in the contour
    if len(approx)==3:
        shape='triangle'
    elif len(approx)==4:
        #compute the bounding box of the contour and find the aspect ratio
        (x,y,w,h)=cv2.boundingRect(approx)
        ar=w/float(h)

        shape='square' if ar >=0.95 and ar <=1.05 else 'rectangle'
    else:
        shape='circle'

    return shape




# def range_invert(array):
#     max_val = np.amax(array)
#     volume = array / max_val
#     min_val = np.amin(volume)
#     volume = volume - min_val
#     volume = (1 - volume)  # inverting the range
#     array = volume * max_val
#
#     return array







def mtf_calc(ROI):
    print('calculating MTF')
    # see doselab manual for method of calculation
    M5num=np.percentile(ROI[len(ROI)-1],90)-np.percentile(ROI[len(ROI)-1],10)
    M5den=np.percentile(ROI[len(ROI)-1],90)+np.percentile(ROI[len(ROI)-1],10)
    M5=M5num/M5den


    LinePairs = [0.76, 0.43, 0.23, 0.20, 0.1]
    MTF=[]
    for region in ROI:
        num=np.percentile(region,90)-np.percentile(region,10)
        den=np.percentile(region,90)+np.percentile(region,10)
        Mi=num/den
        MTF.append(Mi/M5)

    print(MTF)


    # plt.figure()
    # plt.plot(LinePairs,MTF)
    # plt.title('rMTF plot')
    # plt.xlabel('Line pairs lp/mm')
    # plt.ylabel('MTF')
    # plt.ylim((0,1))
    # plt.xlim((0.1,0.76))
    # plt.show(block=False)


    # we also want to integrate the MTF and we use the Newton-Cotes formula for integration over a irregular space
    iMTF=abs(np.trapz(MTF,LinePairs)) #since the x is reversed (higher to lower) we just the the absolute value
    print('integral MTF',iMTF)

    return MTF,iMTF




def cnr_calc(ROI,ROInoise):
    print('calculating CNR')

    mean_0=np.mean(ROI[0])
    mean_1=np.mean(ROI[1])

    contrast = 100 * abs(mean_0-mean_1)/(mean_0+mean_1)
    print('contrast=', contrast)

    # std_dev_noise_0=np.std(ROI[0])
    std_dev_noise_0=np.std(ROInoise)

    # std_dev_noise_0=np.std(ROInoise[0])
    # std_dev_noise_1=np.std(ROInoise[1])
    # std_dev_noise_2=np.std(ROInoise[2])
    # std_dev_noise_3=np.std(ROInoise[3])
    # std_dev_noise_4=np.std(ROInoise[4])
    # std_dev_noise_5=np.std(ROInoise[5])
    # avg_std_dev_noise=(1/sqrt(2))*((std_dev_noise_0+std_dev_noise_1+std_dev_noise_2+std_dev_noise_3+std_dev_noise_4+std_dev_noise_5)/6)
    # print('avg_std_dev_noise=',avg_std_dev_noise)

    print('std_dev_noise_0=',std_dev_noise_0)





    # noise= 100*sqrt(std_dev_noise_0*std_dev_noise_0+std_dev_noise_1*std_dev_noise_1)/sqrt(mean_0*mean_0+mean_1+mean_1)
    # print('noise=',noise)


    # cnr = contrast/noise
    # cnr = abs(mean_0-mean_1)/avg_std_dev_noise

    # cnr using only the std_dev of the dark region (ROI11)
    cnr = abs(mean_0-mean_1)/std_dev_noise_0


    print('cnr=',cnr)


    textstr='Random Noise='+'{:4f}'.format(float(std_dev_noise_0)) + '\n' + 'CNR=' + '{:4f}'.format(float(cnr))+'\n'


    return textstr




























# def read_dicom(filename1,filename2,ioption):
def read_dicom(dirname,ioption):
    titletype=['LDR','HDR','setup']
    for subdir, dirs, files in os.walk(dirname):
        for file in tqdm(sorted(files)):
            if os.path.splitext(dirname + file)[1] == '.dcm':
                dataset = pydicom.dcmread(dirname + file)
                station_name=dataset[0x3002,0x0020].value

                if dataset[0x300c, 0x0006].value==1:
                    LDRDicom = np.zeros((dataset.Rows, dataset.Columns, 0), dtype=dataset.pixel_array.dtype)
                    tmp_array = dataset.pixel_array
                    LDRDicom = np.dstack((LDRDicom, tmp_array))
                    # print("slice thickness [mm]=", dataset.SliceThickness)
                    SID = dataset.RTImageSID
                    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
                    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
                    print("pixel spacing row [mm]=", dx)
                    print("pixel spacing col [mm]=", dy)
                elif dataset[0x300c, 0x0006].value==2:
                    tmp_array = dataset.pixel_array
                    LDRDicom = np.dstack((LDRDicom, tmp_array))
                    # titletype.append('LDR')


                if dataset[0x300c, 0x0006].value==3:
                    HDRDicom = np.zeros((dataset.Rows, dataset.Columns, 0), dtype=dataset.pixel_array.dtype)
                    tmp_array = dataset.pixel_array
                    HDRDicom = np.dstack((HDRDicom, tmp_array))
                    # print("slice thickness [mm]=", dataset.SliceThickness)
                    SID = dataset.RTImageSID
                    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
                    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
                    print("pixel spacing row [mm]=", dx)
                    print("pixel spacing col [mm]=", dy)
                elif dataset[0x300c, 0x0006].value==4:
                    tmp_array = dataset.pixel_array
                    HDRDicom = np.dstack((HDRDicom, tmp_array))
                    # titletype.append('HDR')

                if dataset[0x300c, 0x0006].value==5:
                    setupDicom = np.zeros((dataset.Rows, dataset.Columns, 0), dtype=dataset.pixel_array.dtype)
                    tmp_array = dataset.pixel_array
                    setupDicom = np.dstack((setupDicom, tmp_array))
                    # print("slice thickness [mm]=", dataset.SliceThickness)
                    SID = dataset.RTImageSID
                    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
                    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
                    print("pixel spacing row [mm]=", dx)
                    print("pixel spacing col [mm]=", dy)
                elif dataset[0x300c, 0x0006].value==6 or dataset[0x300c, 0x0006].value==12:
                    tmp_array = dataset.pixel_array
                    setupDicom = np.dstack((setupDicom, tmp_array))
                    # titletype.append('setup')

    print(titletype)
    # exit(0)

    # print(np.shape(setupDicom),np.shape(HDRDicom),np.shape(LDRDicom))

    ArrayDicom = LDRDicom

    ArrayDicom = np.dstack((LDRDicom,HDRDicom))
    ArrayDicom = np.dstack((ArrayDicom,setupDicom))
    if np.shape(ArrayDicom)[2]<6:
        print('Data incomplete. Exiting....')
        exit(0)

    print(np.shape(ArrayDicom),np.shape(ArrayDicom)[2]//2,titletype)


    figs=[]
    mtf_fig = plt.figure(figsize=(7, 5))
    ax = mtf_fig.subplots()
    for i in range(0,np.shape(ArrayDicom)[2]//2):
        LinePairs = [0.76, 0.43, 0.23, 0.20, 0.1]
        print(2*i,2*i+1)
        data_o=ArrayDicom[:,:,2*i]
        data_1=ArrayDicom[:,:,2*i]
        data_2=ArrayDicom[:,:,2*i+1]

        # test to make sure image is displayed correctly bibs are high amplitude against dark background
        ctr_pixel = data_1[np.shape(data_1)[0] // 2, np.shape(data_1)[1] // 2]
        corner_pixel = data_1[0, 0]

        if ctr_pixel < corner_pixel:  # we need to invert the image range for both clinacs and tb
            data_1 = u.range_invert(data_1)
            data_2 = u.range_invert(data_2)

        rand_noise = data_1 - data_2  # we need the random noise so we can calculate the MTF function
        # ArrayDicom_f= cv2.bilateralFilter(np.asarray(ArrayDicom,dtype='float32'), 33, 33, 17) #aggresive
        # ArrayDicom_f= cv2.bilateralFilter(np.asarray(ArrayDicom,dtype='float32'), 27, 27, 17) #good with (method2 mean0-mean1)/std_ROInoise[1]
        data_1_f = cv2.bilateralFilter(np.asarray(data_1, dtype='float32'), 3, 17, 17)  # mild

        data_1 = u.norm01(data_1)
        data_2 = u.norm01(data_2)

        data_1 = 255 * data_1
        data_2 = 255 * data_2
        # print(ArrayDicom.dtype)

        data_1 = data_1.astype(np.uint8)
        data_1 = data_2.astype(np.uint8)

        # performing bilateral filtering to remove some noise without affecting the edges
        img_bifilt = cv2.bilateralFilter(data_1, 11, 17, 17)
        # edged=cv2.Canny(img_bifilt,30,200)

        # #thresholding the image to find the square
        th = cv2.threshold(img_bifilt, 200, 255, cv2.THRESH_TRUNC)[1]
        th2 = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        th2 = cv2.bitwise_not(th2)

        # doing blob detection
        # blobs_log = blob_log(th2, min_sigma=3, max_sigma=5, num_sigma=20, threshold=0.5,exclude_border=True)
        blobs_log = blob_log(th2, min_sigma=3, max_sigma=5, num_sigma=20,
                             threshold=0.5)  # run on windows, for some stupid reason exclude_border is not recognized in my distro at home

        center = []
        point_det = []
        for blob in blobs_log:
            y, x, r = blob
            point_det.append((x, y, r))

        point_det = sorted(point_det, key=itemgetter(2), reverse=True)

        for j in range(0, 2):
            x, y, r = point_det[j]
            center.append((int(round(x)), int(round(y))))




        # Now that we have correctly detected the points we need to estimate the scaling of the image and the location of every ROI
        x1, y1 = center[0]
        x2, y2 = center[1]

        theta = atan(abs(y2 - y1) / abs(x2 - x1))
        theta_deg = degrees(theta)

        print('theta_deg (angle the phantom was placed) =', theta_deg)

        # The distance between the centers of the ROIs in pixels is given by
        dist_horz_roi = int(
            21 / dx)  # where 21 mm is the witdh of the ROI each ROI is 20 mm width with 1mm spacer and 28mm in height
        dist_vert_roi = int(
            28 / dy)  # where 21 mm is the witdh of the ROI each ROI is 20 mm width with 1mm spacer and 28mm in height
        width_roi = int(20 / dx) - 10  # just subtracting a few pixels to avoid edge effects
        height_roi = int(27 / dy) - 10
        print('dist_horz_roi=', dist_horz_roi)



        # The ROIs location can be identified by its positions with respect to the two points
        # let's rotate the image around the center of the first ROI
        xrot = int(abs(x2 + x1) / 2)
        yrot = int(abs(y2 + y1) / 2)



        M = cv2.getRotationMatrix2D((xrot, yrot), theta_deg, 1)
        data_o_rot = cv2.warpAffine(u.range_invert(data_o), M, (np.shape(data_o)[1], np.shape(data_o)[0]))  # if we want to use the filtered values
        data_f_rot = cv2.warpAffine(data_1_f, M, (np.shape(data_1_f)[1], np.shape(data_1_f)[0]))  # if we want to use the filtered values
        # ArrayDicom_rot=cv2.warpAffine(ArrayDicom,M,(np.shape(ArrayDicom_o)[1],np.shape(ArrayDicom_o)[0]))
        rand_noise_rot = cv2.warpAffine(rand_noise, M, (np.shape(rand_noise)[1], np.shape(rand_noise)[0]))



        ROImtf = []
        ROImtf.append(data_o_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
                      xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
        ROImtf.append(data_o_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
                      xrot - dist_horz_roi - int(width_roi / 2):xrot - dist_horz_roi + int(width_roi / 2)])
        ROImtf.append(data_o_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
                      xrot + dist_horz_roi - int(width_roi / 2):xrot + dist_horz_roi + int(width_roi / 2)])
        ROImtf.append(data_o_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
                      xrot - 2 * dist_horz_roi - int(width_roi / 2):xrot - 2 * dist_horz_roi + int(width_roi / 2)])
        ROImtf.append(data_o_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
                      xrot + 2 * dist_horz_roi - int(width_roi / 2):xrot + 2 * dist_horz_roi + int(width_roi / 2)])

        ROIcnr = []
        ROIcnr.append(
            data_o_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
            xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
        ROIcnr.append(
            data_o_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
            xrot - int(width_roi / 2):xrot + int(width_roi / 2)])

        ROIcnr_noise = []
        ROIcnr_noise.append(
            rand_noise_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
            xrot - int(width_roi / 2):xrot + int(width_roi / 2)])

        # ROIcnr_noise.append(rand_noise_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
        #               xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
        #
        # ROIcnr_noise.append(rand_noise_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
        #               xrot - dist_horz_roi- int(width_roi / 2):xrot - dist_horz_roi+ int(width_roi / 2)])
        # ROIcnr_noise.append(rand_noise_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
        #               xrot - dist_horz_roi- int(width_roi / 2):xrot - dist_horz_roi+ int(width_roi / 2)])
        #
        # ROIcnr_noise.append(rand_noise_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
        #               xrot + dist_horz_roi- int(width_roi / 2):xrot + dist_horz_roi + int(width_roi / 2)])
        # ROIcnr_noise.append(rand_noise_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
        #               xrot + dist_horz_roi- int(width_roi / 2):xrot + dist_horz_roi + int(width_roi / 2)])

        # now that we have the ROIs we can proceed to calculate the rMTF
        MTF,iMTF=mtf_calc(ROImtf)
        print(titletype,'MTF=',MTF)



        # now that we have the ROIs we can proceed to calculate the CNR (contrast to noise ratio and the random noise)
        textstr = cnr_calc(ROIcnr, ROIcnr_noise)
        textstr=titletype[i]+'\n'+'Unit='+str(station_name)+'\n'+textstr+'Integrated MTF='+ '{:4f}'.format(float(iMTF))
        print('this is the text string',textstr)


        figs.append(viewer(data_1, dx, dy, center, titletype[i],textstr))


        # creating the MTF figure
        ax.set_title('rMTF plot')
        ax.set_xlabel('Line pairs lp/mm')
        ax.set_ylabel('MTF')
        ax.set_ylim((0, 1))
        ax.set_xlim((0.1, 0.76))
        ax.hlines(0.5,0.1,0.76,colors='r',linestyles='--')
        ax.plot(LinePairs, MTF,label=titletype[i])
        ax.legend()






    with PdfPages(dirname + '/' + 'Epid_report.pdf') as pdf:
    # with PdfPages('Epid_report.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig)

        pdf.savefig(mtf_fig)

    # plt.show()
    exit(0)




    # # Normal mode:
    # print()
    # print("Filename.........:", file)
    # print("Storage type.....:", dataset.SOPClassUID)
    # print()
    #
    # pat_name = dataset.PatientName
    # display_name = pat_name.family_name + ", " + pat_name.given_name
    # print("Patient's name...:", display_name)
    # print("Patient id.......:", dataset.PatientID)
    # print("Modality.........:", dataset.Modality)
    # print("Study Date.......:", dataset.StudyDate)
    # print("Gantry angle......", dataset.GantryAngle)
    # #
    # # if 'PixelData' in dataset:
    # #     rows = int(dataset.Rows)
    # #     cols = int(dataset.Columns)
    # #     print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
    # #         rows=rows, cols=cols, size=len(dataset.PixelData)))
    # #     if 'PixelSpacing' in dataset:
    # #         print("Pixel spacing....:", dataset.PixelSpacing)
    # #
    # # # use .get() if not sure the item exists, and want a default value if missing
    # # print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('direpid',type=str,help="Input the directory name")

# parser.add_argument('epid1',type=str,help="Input the filename")
# parser.add_argument('epid2',type=str,help="Input the filename")
# parser.add_argument('-a', '--add', nargs='?', type=argparse.FileType('r'), help='additional file for averaging before processing')
args=parser.parse_args()

dirname=args.direpid

# filename1=args.epid1
# filename2=args.epid2





while True:  # example of infinite loops using try and except to catch only numbers
    line = input('Are these files from a clinac [yes(y)/no(n)]> ')
    try:
        ##        if line == 'done':
        ##            break
        ioption = str(line.lower())
        if ioption.startswith(('y', 'yeah', 'yes', 'n', 'no', 'nope')):
            break

    except:
        print('Please enter a valid option:')




# read_dicom(filename1,filename2,ioption)
read_dicom(dirname,ioption)
