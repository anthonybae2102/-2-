import os
import zipfile
import fnmatch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import staintools
import random
import h5py
import time

sub_folder_no = 0
file_no = []
saved_folder = []
RAW_IM_DIRECTORY = '/share_folder/data/HBS_cropped_images/crop_images'
NORM_IM_DIRECTORY = '/share_folder/data/HBS_cropped_images/norm_images'

#prepare target image as benchmark for stain normalization
target_dir = '/share_folder/data/HBS_cropped_images'
target = cv.imread(os.path.join(target_dir, 'standard_histo_image.png'))#standard image for color norm
target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
target = staintools.LuminosityStandardizer.standardize(target)

for f in os.listdir(RAW_IM_DIRECTORY):
    if f not in saved_folder:
        
        saved_folder.append(f)#record save history
        if sub_folder_no%5 ==0:
            start_time = time.time()

        sub_folder_no += 1
        count_file = 0

        SUB_DIRECTORY = os.path.join(RAW_IM_DIRECTORY, f)
        NORM_SUB_DIRECTORY = os.path.join(NORM_IM_DIRECTORY, f)
        os.mkdir(NORM_SUB_DIRECTORY)

        for filename in os.listdir(SUB_DIRECTORY):
            count_file += 1        
            img = cv.imread(os.path.join(SUB_DIRECTORY, filename))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # stain normalization starts
            to_transform = img

            # Standardize brightness (This step is optional but can improve the tissue mask calculation)
            to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

            # Stain normalize
            normalizer = staintools.StainNormalizer(method='vahadane')
            normalizer.fit(target)
            norm_img = normalizer.transform(to_transform)

            #save norm image
            norm_file_name = os.path.join(NORM_SUB_DIRECTORY, filename)
            cv.imwrite(norm_file_name, cv.cvtColor(norm_img, cv.COLOR_BGR2RGB))


        file_no.append(count_file)
        if sub_folder_no%5 == 0:
            print(f'current no. of images normalized is {np.sum(file_no)}')
            end_time = time.time()
            print(f'Elapsed time image is {end_time-start_time}')
     
    else:
        print('same folder is being saved')



