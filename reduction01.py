import sys,os
import numpy as np
import os
from astropy.io import fits
from datetime import datetime, timedelta

def findprefix(i):
    #function to find prefix of the ith event files from raw data
    if i<10:
        prefix = "event00" + str(i)
    else:
        prefix = "event0" + str(i)
    return prefix
    
def time_avg(timestamps):
    """
    Function will find an average of timestamps in list in YSO format and return string 
    """
    # Convert strings to datetime objects
    datetime_objects = [datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f") for ts in timestamps]

    # Convert all times to seconds since the first timestamp
    time_deltas = [(dt - datetime_objects[0]).total_seconds() for dt in datetime_objects]

    # Compute the average of the time deltas
    average_seconds = sum(time_deltas) / len(time_deltas)

    # Compute the average datetime
    average_datetime = datetime_objects[0] + timedelta(seconds=average_seconds)

    # Convert back to string with milliseconds
    average_timestamp = average_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
    return average_timestamp


def average_fits_files(fits_dir, i, output_path):
    """
    Averages multiple FITS files containing RA, Dec, and photon counts.

    Parameters:
    - fits_dir (str): Path to the directory containing FITS files
    - ith event to be averaged
    - output_path: it is where you want output to be stored
    - returns output_filename (str): Name of the output FITS file
    - all raw files have extension ".fit" but processed files from hereon will have ".fits"

    """
    prefix = findprefix(i)
    fits_files = [os.path.join(fits_dir, f) for f in os.listdir(fits_dir) if f.startswith(prefix) and f.endswith(".fit")]

    data_list = []
    j = 0 #index of each file to be averaged
    avg_date = []
    avg_dateobs = []
    
    # Read and store data from each FITS file
    for file in fits_files:
        with fits.open(file) as hduj:
            data_list.append(hduj[0].data)
            j += 1
            #append DATE and DATE-OBS from header file
            avg_date.append(hduj[0].header['DATE'])
            avg_dateobs.append(hduj[0].header['DATE-OBS'])
            
    print(j)
    # Convert to numpy array and compute the mean
    data_stack = np.array(data_list)
    averaged_data = np.mean(data_stack, axis=0)

    # Setting header from the first FITS file
    hdr = fits.getheader(fits_files[0])
    
    #Everything else for the header
    hdr['DATE'] = time_avg(avg_date)
    hdr['DATE-OBS'] = time_avg(avg_dateobs)
    hdr['NFRAMES'] = j
    print(time_avg(avg_date))
    
    # Defining averaged filename
    filename = prefix + "_" + str(hdr['EXPOSURE']) + "_" + time_avg(avg_dateobs) + ".fits"

    # Write the averaged image to a new FITS file
    new_file = os.path.join(output_path,filename)
    fits.writeto(new_file, averaged_data, hdr, overwrite=True)

    return new_file

fits_dir = "/home/manya/Documents/PHD2_CameraFrames_2025-02-28-190953/"
output_path = "/home/manya/Documents/Reduction1/"
average_fits_files(fits_dir, 1, output_path)

