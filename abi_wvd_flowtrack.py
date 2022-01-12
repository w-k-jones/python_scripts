#!/home/users/wkjones/miniconda2/envs/py3/bin/python3.7
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
import cv2 as cv

from python_toolbox import abi_tools, tracking_tools, dataset_tools

features_filename = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/abi_RadC_features_S2018-06-19T12:00:00_E2018-06-20T00:00:00.npz'
with np.load(features_filename, allow_pickle=True, encoding='latin1') as features_file:
    labels = features_file['labels']
    label_files = features_file['files']

with xr.open_mfdataset([f[8] for f in label_files], combine='nested', concat_dim='t') as ds:
    BT_ch8 = abi_tools.get_abi_IR(ds)

with xr.open_mfdataset([f[10] for f in label_files], combine='nested', concat_dim='t') as ds:
    BT_ch10 = abi_tools.get_abi_IR(ds)

dataset_tools.match_coords([BT_ch8, BT_ch10], 't')

wvd = BT_ch8 - BT_ch10

linked_labels = np.zeros(wvd.shape)

linked_labels[:2] = tracking_tools.optical_flow_track(wvd[0], wvd[1], labels[0], labels[1])

for i in range(1,label_files.shape[0]-1):
    linked_labels[i:i+2] = tracking_tools.optical_flow_track(wvd[i], wvd[i+1], linked_labels[i], labels[i+1])

linked_labels = linked_labels.astype('uint16')

unique_labels = np.unique(linked_labels).astype('uint16')

linked_labels = np.digitize(linked_labels, unique_labels).astype('uint16')

savename = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/abi_RadC_linked_S2018-06-19T12:00:00_E2018-06-20T00:00:00.npz'
np.savez(savename, files=label_files, linked_labels=linked_labels)
