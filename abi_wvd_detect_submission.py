#!/home/users/wkjones/miniconda2/envs/NEW/bin/python2.7
from __future__ import print_function, division
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import argparse
import os

parser = argparse.ArgumentParser(description="""Detect deep convective features
    in GOES-16 ABI imagery using a combination of wvd, edge detection and "3d" methods""")
parser.add_argument('satellite', help='GOES satellite to use (16 or 17)', type=int)
parser.add_argument('start_date', help='Start date of processing', type=str)
parser.add_argument('end_date', help='End date of processing', type=str)
parser.add_argument('chunk_hours', help='Hours to process per submitted job', type=int)
parser.add_argument('overlap_hours', help='Hours to overlap each job by', type=int)
parser.add_argument('-C', help='Use CONUS scan (5 minute) data', action='store_true', default=False)
parser.add_argument('-F', help='Use Full scan (15 minute) data', action='store_true', default=False)

args = parser.parse_args()

if args.C:
    scan_type = '-C'
elif args.F:
    scan_type = '-F'
else:
    raise ValueError("""Error in abi_wvd_detect_submission: -C or -F input must be selected""")

satellite = str(args.satellite)
start_date = parse_date(args.start_date)
end_date = parse_date(args.end_date)
print(args.chunk_hours)
print(args.overlap_hours)
if args.overlap_hours >= args.chunk_hours:
    raise ValueError("""Error in abi_wvd_detect_submission: overlap_hours argument
                        must be less than chunk_hours""")
print('Job submission for satellite GOES'+satellite)
print('Start date:', start_date.isoformat())
print('End date:', end_date.isoformat())

offset_hours = args.chunk_hours - args.overlap_hours
offset_td = timedelta(hours=offset_hours)
chunk_td = timedelta(hours=args.chunk_hours)

submit_start_date = parse_date(args.start_date)
submit_end_date = submit_start_date + chunk_td

submit_dates = [(submit_start_date, submit_end_date)]

while submit_end_date < end_date:
    submit_start_date += offset_td
    submit_end_date = submit_start_date + chunk_td
    submit_dates.append((submit_start_date, submit_end_date))

print('Chunks created:', len(submit_dates))

print('Starting job submission')
bsub_str = 'bsub -q short-serial -W 04:00 -R "rusage[mem=64000]" -M 64000'
pyfile_str = '/home/users/wkjones/python/abi_wvd_detect.py'
for n, dates in enumerate(submit_dates):
    job_name = 'abi_Rad'+scan_type[1]+'_features_S'+dates[0].isoformat()+'_E'+dates[1].isoformat()
    process_str = ' '.join([pyfile_str, satellite, dates[0].isoformat(), dates[1].isoformat(), scan_type])
    exec_str = ' '.join([bsub_str, '-oo', job_name+'.out', '-eo', job_name+'.err', '-J', job_name, process_str])
    print('Submitting job:', n)
    #print(exec_str)
    os.system(exec_str)
