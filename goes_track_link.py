#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing
from skimage.morphology import watershed
from scipy.ndimage.filters import sobel, gaussian_filter
from scipy.ndimage import label, find_objects
from datetime import datetime, timedelta, date, time
from skimage.feature import peak_local_max
from pyproj import Proj
import argparse


def get_abi_lat_lon(dataset):
    p = Proj(proj='geos', h=dataset.goes_imager_projection.perspective_point_height,
             lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
             sweep=dataset.goes_imager_projection.sweep_angle_axis)
    xx, yy = np.meshgrid(dataset.x.data*dataset.goes_imager_projection.perspective_point_height,
                         dataset.y.data*dataset.goes_imager_projection.perspective_point_height)
    lons, lats = p(xx, yy, inverse=True)
    lons[lons>=1E30] = np.nan
    lats[lats>=1E30] = np.nan
    return lats, lons

def get_abi_pixel_area(dataset):
    lat, lon = get_abi_lat_lon(dataset)
    nadir_res = float(dataset.spatial_resolution.split('km')[0])
    xx, yy = np.meshgrid(dataset.x.data,
                         dataset.y.data)
    lx_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.longitude_of_projection_origin-lon))+np.abs(xx))
    ly_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.latitude_of_projection_origin-lat))+np.abs(yy))
    area = nadir_res**2/(lx_factor*ly_factor)
    return area

def get_abi_IR(dataset):
    planck_shape = dataset.planck_fk2.shape
    planck_ndims = len(planck_shape)
    data_shape = dataset.Rad.shape
    data_ndims = len(data_shape)
    planck_reshape = planck_shape + (1,)*(data_ndims-planck_ndims)
    bt = (dataset.planck_fk2.data.reshape(planck_reshape) / (np.log((dataset.planck_fk1.data.reshape(planck_reshape) / dataset.Rad.data) + 1)) - dataset.planck_bc1.data.reshape(planck_reshape)) / dataset.planck_bc2.data.reshape(planck_reshape)
    DQF = dataset.DQF.data
    bt[DQF<0] = np.nan
    bt[DQF>1] = np.nan
    return bt

def get_abi_ref(dataset):
    ref = dataset.Rad.data * dataset.kappa0.data
    DQF = dataset.DQF.data
    ref[DQF<0] = np.nan
    ref[DQF>1] = np.nan
    return ref

def get_wvd_from_files(C8_file, C10_file):
    with xr.open_dataset(C8_file) as C8_ds:
        C8_BT = get_abi_IR(C8_ds)
    with xr.open_dataset(C10_file) as C10_ds:
        C10_BT = get_abi_IR(C10_ds)
    wvd = C8_BT - C10_BT
    return wvd

def sobel_2d(wvd):
    edges = np.sqrt(sobel(wvd,1)**2 + sobel(wvd,0)**2)
    return edges

def sobel_3d(wvd):
    edges = np.sqrt(sobel(wvd,1)**2 + sobel(wvd,2)**2 + sobel(wvd,0)**2)
    return edges

def mark_threshold(wvd, background=-10, threshold=-1):
    markers = np.full_like(wvd, 0, dtype='uint8')
    markers += 2*(wvd>threshold).astype('uint8')
    markers += grey_erosion((wvd<=background), size=(11,11)).astype('uint8')
    return markers

def get_area_max(x_in, l=1, nanmax=False):
    x = x_in[:(x_in.shape[0]//l)*l,:(x_in.shape[1]//l)*l]
    x = x.reshape(x_in.shape[0]//l, l, x_in.shape[1]//l, l)
    x = np.moveaxis(x, 1, 2).reshape(x_in.shape[0]//l, x_in.shape[1]//l, l**2)
    if nanmax:
        x = np.nanmax(x, axis=2)
    else:
        x = x.max(2)
    return x

def get_goes_abi_files(input_file):
    datestr = input_file.split('/')[-1].split('_')[3]
    yearstr = datestr[1:5]
    doystr = datestr[5:8]
    hourstr = datestr[8:10]
    minstr = datestr[10:12]
    secstr = datestr[12:14]
    file_date = datetime(year=int(yearstr), month=1, day=1, hour=int(hourstr), minute=int(minstr), second=int(secstr)) + timedelta(days=int(doystr)-1)
    files = glob('/'.join(input_file.split('/')[:-1])+'/'+input_file.split('/')[-1].split('_')[0][:-2]+'*s'+yearstr+doystr+hourstr+minstr+'*.nc')
    return [file_date]+files

def get_start_end_index(files):
    file_date = files[0][0]
    if file_date.hour < 12:
        start_date = datetime.combine(file_date.date(), time(hour=6))
        end_date = datetime.combine(file_date.date(), time(hour=18))
    else:
        start_date = datetime.combine(file_date.date(), time(hour=18))
        end_date = datetime.combine((file_date+timedelta(days=1)).date(), time(hour=6))
    start_ind = np.arange(len(files))[[f[0]>start_date for f in files]][0]
    end_ind = np.arange(len(files))[[f[0]<end_date for f in files]][-1]
    return start_ind, end_ind, start_date, end_date

def get_BT_stats(input_files, slices, features):
    files = input_files[slices[0]]
    shape = tuple([s.stop-s.start for s in slices])
    BT_array = ma.array(np.empty(shape), mask=np.logical_not(features))
    for i,f in enumerate(files):
        with xr.open_dataset(f) as C14:
            BT_array.data[i] = get_abi_IR(C14)[slices[1:]]
    return(BT_array)

def merge_slices(slice_list):
    starts = [s.start for s in slice_list]
    start = min(starts)
    ends = [s.stop for s in slice_list]
    end = max(ends)
    merged_slice = slice(start, end)
    new_len = end-start
    offsets = [s-start for s in starts]
    return merged_slice, new_len, offsets

def merge_slices_nd(slice_list_nd):
    nd = len(slice_list_nd[0])
    slices_1d = [merge_slices([s[i] for s in slice_list_nd]) for i in range(nd)]
    merged_slices = [s[0] for s in slices_1d]
    new_shape = tuple([s[1] for s in slices_1d])
    offsets = [[slices_1d[i][2][j] for i in range(nd)] for j in range(len(slice_list_nd))]
    return merged_slices, new_shape, offsets

def recursive_linker(links_list1=None, links_list2=None, label_list1=None, label_list2=None, overlap_list1=None, overlap_list2=None):
    recursive = False
    if links_list1 is None:
        links_list1=[]
    if links_list2 is None:
        links_list2=[]
    if label_list1 is None:
        label_list1=[]
    if label_list2 is None:
        label_list2=[]
    if overlap_list1 is None:
        overlap_list1=[]
    if overlap_list2 is None:
        overlap_list2=[]
    for i in links_list1:
        if i in label_list1:
            loc = label_list1.index(i)
            label = label_list1.pop(loc)
            overlap = overlap_list1.pop(loc)
            for j in overlap:
                if j not in links_list2:
                    links_list2.append(j)
                    recursive = True
    if recursive:
        links_list2, links_list1 = recursive_linker(links_list1=links_list2, links_list2=links_list1, label_list1=label_list2, label_list2=label_list1, overlap_list1=overlap_list2, overlap_list2=overlap_list1)
    return links_list1, links_list2

def link_labels(labels1, labels2):
    label_list1 = np.unique(labels1[labels1!=0]).tolist()
    label_list2 = np.unique(labels2[labels2!=0]).tolist()
    overlap_list1 = [np.unique(labels2[np.logical_and(labels2>0,labels1==i)]).tolist() for i in label_list1]
    overlap_list2 = [np.unique(labels1[np.logical_and(labels1>0,labels2==j)]).tolist() for j in label_list2]
    links_list1 = []
    links_list2 = []
    while len(label_list1)>0:
        temp_links1, temp_links2 = recursive_linker([label_list1[0]], label_list1=label_list1, label_list2=label_list2, overlap_list1=overlap_list1, overlap_list2=overlap_list2)
        links_list1.append(temp_links1)
        links_list2.append(temp_links2)
    return links_list1, links_list2

def get_object_info(object_slice, files_list, feature_mask, lats, lons):
    object_info={}
    object_info['files']=files_list[object_slice[0]]
    object_info['slice']=object_slice[1:]
    object_info['duration'] = (object_info['files'][-1][0] - object_info['files'][0][0]).total_seconds()
    y0 = object_slice[1].start
    y1 = object_slice[1].stop-1
    x0 = object_slice[2].start
    x1 = object_slice[2].stop-1
    object_info['UL_corner_latlon'] = [lats[y0,x0],lons[y0,x0]]
    object_info['LL_corner_latlon'] = [lats[y1,x0],lons[y1,x0]]
    object_info['UR_corner_latlon'] = [lats[y0,x1],lons[y0,x1]]
    object_info['LR_corner_latlon'] = [lats[y1,x1],lons[y1,x1]]
    object_info['feature_mask'] = feature_mask[object_slice]
    return object_info

def merge_file_lists(files_list):
    C14_files = [f[14] for files in files_list for f in files]
    C14_files = np.unique(C14_files).tolist()
    C14_files.sort()
    new_files = [get_goes_abi_files(f) for f in C14_files]
    offsets = [C14_files.index(f[0][14]) for f in files_list]
    return new_files, offsets

def merge_object_info(object_info_list, lats, lons):
    object_info={}
    files, file_offsets = merge_file_lists([info['files'] for info in object_info_list])
    object_info['files'] = files
    object_info['duration'] = (object_info['files'][-1][0] - object_info['files'][0][0]).total_seconds()
    merged_slices = merge_slices_nd([info['slice'] for info in object_info_list])
    object_info['slice'] = merged_slices[0]
    y0 = merged_slices[0][0].start
    y1 = merged_slices[0][0].stop-1
    x0 = merged_slices[0][1].start
    x1 = merged_slices[0][1].stop-1
    object_info['UL_corner_latlon'] = [lats[y0,x0],lons[y0,x0]]
    object_info['LL_corner_latlon'] = [lats[y1,x0],lons[y1,x0]]
    object_info['UR_corner_latlon'] = [lats[y0,x1],lons[y0,x1]]
    object_info['LR_corner_latlon'] = [lats[y1,x1],lons[y1,x1]]
    feature_mask = np.full((len(files),)+merged_slices[1], False)
    for i, info in enumerate(object_info_list):
        shape = info['feature_mask'].shape
        soff = merged_slices[2][i]
        foff = file_offsets[i]
        feature_mask[foff:foff+shape[0], soff[0]:soff[0]+shape[1], soff[1]:soff[1]+shape[2]] = np.logical_or(feature_mask[foff:foff+shape[0], soff[0]:soff[0]+shape[1], soff[1]:soff[1]+shape[2]],info['feature_mask'])
    object_info['feature_mask'] = feature_mask
    return object_info

parser = argparse.ArgumentParser(description='Detect deep convective features in GOES-16 ABI imagery')
parser.add_argument('start', help='Day of year to start process', type=int)
parser.add_argument('end', help='Day of year to end process', type=int)

args = parser.parse_args()

features_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/features/'
save_path = '/gws/nopw/j04/eo_shared_data_vol2/scratch/satellite/GOES16/objects/'
filename = glob(features_path+'RadC_features_'+str(args.start).zfill(3)+'.npz')
if len(filename) == 0:
    files = ['']
else:
    files = [filename[0]]
# Get files in backwards order
for i in range(args.start,args.end-1,-1):
    doy_str = str(i).zfill(3)
    hd_filename = glob(features_path+'RadC_features_'+doy_str+'hd.npz')
    if len(hd_filename) == 0:
        files.append('')
    else:
        files.append(hd_filename[0])
    filename = glob(features_path+'RadC_features_'+doy_str+'.npz')
    if len(filename) == 0:
        files.append('')
    else:
        files.append(filename[0])

with np.load(files[0]) as f:
    file_list = f['files']

# Get lat/lon from first file (should be the same for all)
with xr.open_dataset(file_list[0][14]) as ds:
    lat, lon = get_abi_lat_lon(ds)

# Start flag defines if there is no previous file to extend from (no overlapping objects considered)
start_flag = True
# Main loop over mask files
for i, filename in enumerate(files):
    if filename == '':
        print('Missing file: ', i)
        start_flag = True
    else:
        print(i)
        print(filename)
        with np.load(filename) as f:
            labels = f['labels']
            file_list = f['files']

        # Find objects returns a tuple of slices for each individual labelled object
        objects = find_objects(labels)
        objects = [o for o in objects if o is not None]

        # Find the numbers of labels than border the edges of the image (considered invalid)
        edge_labels = np.unique(labels[:,0,:]).tolist() + np.unique(labels[:,-1,:]).tolist() + np.unique(labels[:,:,0]).tolist() + np.unique(labels[:,:,-1]).tolist()
        edge_labels = np.unique(edge_labels)
        edge_labels = edge_labels[edge_labels!=0].tolist()
        try:
            # Finds the 'offical' start and end time of the file +/-6hrs offset, no longer needed as this is in the file name
            # and we want to find the exact overlap anyway
            start, end, sd, ed = get_start_end_index(file_list)
        except:
            print('Invalid due to time gaps: ', filename)
            start_flag = True
        else:
            # Get the datetimes from the list of files, calculate the time difference between each file. Discard if over 15 minutes
            file_times = [f[0] for f in file_list]
            file_timedeltas = [(file_times[i]-file_times[i-1]).total_seconds() for i in range(1,len(file_times))]
            if max(file_timedeltas[start:end]) > 901:
                print('Invalid due to time gaps: ', filename)
                start_flag = True
            else:
                # Get object info for objects that only exist in one file (outside of overlaps)
                short_objects = {(i+1):get_object_info(o, file_list, labels, lat, lon) for i, o in enumerate(objects) if o[0].start > start and o[0].stop < end and (i+1) not in edge_labels}
                for i, info in short_objects.iteritems():
                    info['feature_mask'] = info['feature_mask'] == i

                print('Initial short objects: ', len(short_objects))

                if start_flag:
                    long_objects= {(i+1):get_object_info(o, file_list, labels, lat, lon) for i,o in enumerate(objects) if o[0].start <= start and o[0].stop < end and o[0].stop > start and (i+1) not in edge_labels}
                    for i, info in long_objects.iteritems():
                        info['feature_mask'] = info['feature_mask'] == i

                    print('Initial long objects: ', len(long_objects))

                    start_flag = False

                    print('Calculating start labels')
                    start_labels = labels[start]
                    for i in np.unique(start_labels).tolist():
                        if i not in long_objects.keys():
                            start_labels[start_labels == i] = 0

                else:
                    if max(file_timedeltas[:start]) > 901:
                        print('Cutting long objects due to time gaps: ', filename)
                        start_flag = True
                    else:
                        print('Finding merged objects')
                        overlap_objects = {(i+1):get_object_info(o, file_list, labels, lat, lon) for i,o in enumerate(objects) if o[0].start < end and o[0].stop >= end and (i+1) not in edge_labels}
                        for i, info in overlap_objects.iteritems():
                            info['feature_mask'] = info['feature_mask'] == i

                        print('Initial overlapping objects: ', len(overlap_objects))

                        end_labels = labels[end]
                        print('Calculating end labels')
                        for i in np.unique(end_labels).tolist():
                            if i not in overlap_objects.keys():
                                end_labels[end_labels == i] = 0

                        print('Calculating links')
                        start_links, end_links = link_labels(start_labels, end_labels)
                        print('Start_links: ', len(start_links))
                        print('End_links: ', len(end_links))
                        print('Merging objects')
                        #merged_objects = {(np.nanmax(labels)+i):merge_object_info([overlap_objects[ind] for ind in end_links[i]]+[long_objects[ind] for ind in start_links[i]], lat, lon) for i in range(len(start_links))}
                        merged_objects = {}
                        label_offset = np.nanmax(labels)+1
                        for i in range(len(start_links)):
                            temp_oobj = [overlap_objects[ind] for ind in end_links[i]]
                            temp_lobj = [long_objects[ind] for ind in start_links[i]]
                            merged_objects[label_offset+i] = merge_object_info(temp_oobj + temp_lobj, lat, lon)
                            for ind in start_links[i]:
                                labels[labels==ind] = label_offset+i
                        print('Final merged objects: ', len(merged_objects))
                        print('Max new ind: ', max(merged_objects.keys()))

                        temp_short_objects = {i:o for i,o in merged_objects.iteritems() if o['files'][0][0] > sd}
                        print('Complete merged objects: ', len(temp_short_objects))
                        temp_long_objects = {i:o for i,o in merged_objects.iteritems() if o['files'][0][0] <= sd}
                        print('Continued merged objects: ', len(temp_long_objects))

                        short_objects.update(temp_short_objects)
                        print('Final short objects: ', len(short_objects))
                        long_objects = {(i+1):get_object_info(o, file_list, labels, lat, lon) for i,o in enumerate(objects) if o[0].start <= start and o[0].stop < end and (i+1) not in edge_labels}
                        print('Initial long objects', len(long_objects))
                        long_objects.update(temp_long_objects)
                        print('Final long objects: ', len(long_objects))

                        print('Calculating start labels')
                        start_labels = labels[start]
                        for i in np.unique(start_labels).tolist():
                            if (i not in long_objects.keys()) and (i not in [l for link in start_links for l in link]):
                                start_labels[start_labels == i] = 0
                        for i in range(len(start_links)):
                            for ind in start_links[i]:
                                start_labels[start_labels == ind] = (label_offset+i)

                print('Filtering short objects')
                short_objects = {k:o for k,o in short_objects.iteritems() if o['duration'] > 900}
                savename = 'RadC_objects_'+filename.split('/')[-1].split('_')[-1]
                print('Saving: '+savename)
                np.savez(save_path+savename, labels=np.array(short_objects.keys()), objects=np.array(short_objects.values()))
