#!/home/users/wkjones/miniconda2/envs/slot_process/bin/python
import os
import subprocess
from glob import glob
from google.cloud import storage
import argparse

from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import pandas as pd

parser = argparse.ArgumentParser(description="""Download GOES16 ABI MCMIP files
    from google cloud storage""")
parser.add_argument('date', help='Start date of processing', type=str)
parser.add_argument('directory', help='Root directory to save file to',
                     type=str)
args = parser.parse_args()
goes_dir = args.directory
if not os.path.isdir(goes_dir):
    os.makedirs(goes_dir)

start_date = parse_date(args.date)
end_date = start_date + timedelta(days=1)
date_list = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime().tolist()

def get_goes_MCMIPC(date, save_dir='./', n_pad=0):
    storage_client = storage.Client()
    goes_bucket = storage_client.get_bucket('gcp-public-data-goes-16')

    s_year = str(date.year).zfill(4)
    doy = (date - datetime(date.year,1,1)).days+1
    s_doy = str(doy).zfill(3)
    s_hour = str(date.hour).zfill(2)

    save_path = goes_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    files = [f.split('/')[-1] for f in
             glob(save_path + 'OR_ABI-L2-MCMIPC-M[36]_G16_s'
                  + s_year + s_doy + s_hour + '*.nc')]

    blobs = goes_bucket.list_blobs(
            prefix='ABI-L2-MCMIPC/'+s_year+'/'+s_doy+'/'+s_hour+'/OR_ABI-L2-MCMIPC-',
            delimiter='/'
            )

    for blob in blobs:
        if blob.name.split('/')[-1] not in files:
            print(blob.name.split('/')[-1])
            blob.download_to_filename(save_path + blob.name.split('/')[-1])

    goes_files = glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')

    if n_pad>0:
        date_pre = date - timedelta(hours=1)
        s_year = str(date_pre.year).zfill(4)
        doy = (date_pre - datetime(date_pre.year,1,1)).days+1
        s_doy = str(doy).zfill(3)
        s_hour = str(date_pre.hour).zfill(2)
        save_path = goes_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        files = [f.split('/')[-1] for f in
                 glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')]
        blobs = list(goes_bucket.list_blobs(
                    prefix='ABI-L2-MCMIPC/'+s_year+'/'+s_doy+'/'+s_hour+'/OR_ABI-L2-MCMIPC-',
                    delimiter='/'
                    ))[-n_pad:]

        for blob in blobs:
            if blob.name.split('/')[-1] not in files:
                print(blob.name.split('/')[-1])
                try:
                    blob.download_to_filename(save_path + blob.name.split('/')[-1])
                except ProtocolError or ChunkedEncodingError:
                    # try again
                    try:
                        blob.download_to_filename(save_path + blob.name.split('/')[-1])
                    except ProtocolError or ChunkedEncodingError:
                        pass

        goes_files = glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')[-n_pad:] + goes_files

        date_next = date + timedelta(hours=1)
        s_year = str(date_next.year).zfill(4)
        doy = (date_next - datetime(date_pre.year,1,1)).days+1
        s_doy = str(doy).zfill(3)
        s_hour = str(date_next.hour).zfill(2)
        save_path = goes_dir + '/' + s_year + '/' + s_doy + '/' + s_hour + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        files = [f.split('/')[-1] for f in
                 glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')]
        blobs = list(goes_bucket.list_blobs(
                    prefix='ABI-L2-MCMIPC/'+s_year+'/'+s_doy+'/'+s_hour+'/OR_ABI-L2-MCMIPC-',
                    delimiter='/'
                    ))[:n_pad]

        for blob in blobs:
            if blob.name.split('/')[-1] not in files:
                print(blob.name.split('/')[-1])
                blob.download_to_filename(save_path + blob.name.split('/')[-1])

        goes_files += glob(save_path + '/OR_ABI-L2-MCMIPC-M[36]_G16_s'+s_year+s_doy+s_hour+'*.nc')[:n_pad]

    return goes_files

for date in date_list:
    temp = get_goes_MCMIPC(date, save_dir=goes_dir, n_pad=0)
