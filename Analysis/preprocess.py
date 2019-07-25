import re
import sys
import uproot
import uproot._connect._pandas
import concurrent.futures

import numpy as np
import pandas as pd

sys.path.insert(0, '/eos/user/m/manoj/Projects/hgcal-electron-pion-classification/Analysis/')
from data import load_data, x_times_e
from observables import *
from helpers import *

error_msg, flag = None, None
try:
    error_msg = "Missing input file"
    input_file = sys.argv[1]

    error_msg = "Missing output file"
    output_file = sys.argv[2]
    
    error_msg = "Output file must be a .h5 file"
    re.findall('\.h5$',output_file)[0]
except IndexError:
    flag = True
    
if flag:
    raise Exception(error_msg)
    
executor = concurrent.futures.ThreadPoolExecutor(16) # threads in pool
chunksize = 10000
columns = ['rechit_layer','rechit_energy','rechit_x','rechit_y','rechit_z']

layers = []
ds_columns = ['beamE','baryX','baryY','baryZ','dR','hits','Etotal','E1to20','E21to28','E1to28',
              'E29to40','E1to10byEtot','E1to2byEtot','E26to28byEtot','E1to20byE21to28']

for i in range (1,41): # all 40 layers columns
    layers.append("E%i"%i)

hits = uproot.open(input_file)['rechitntupler/hits'] # assuming every file has this key.
no_of_events = len(hits['event'].array())
beamE = hits['beamEnergy'].array()[0]
start = 0
dataset = pd.DataFrame(columns = ds_columns + layers) # empty dataframe

print("Beam Energy: %i"%beamE)
print("Processing !")
dots = "."* int(no_of_events/chunksize + 1)
while start < no_of_events:
    print(dots)
    dots = dots[:-1]
    end = start + chunksize
    if end > no_of_events:
        end = start + no_of_events % chunksize

    df = hits.pandas.df(columns, entrystart = start,entrystop = end,
                        flatten = True,executor = executor)
    start = end

    sel = (df.rechit_energy > 0.50)
    df = df[sel]

    df.dropna(subset=['rechit_energy']) # drop missing rows
    x_times_e(df) # add x * energy in dataset

    df = df.reset_index(level=1,drop=True)
    df.index.name = 'event'

    _temp = pd.DataFrame(columns = ds_columns) # emplty dataframe

    _temp.baryX = get_barycenter(df,'x')
    _temp.baryY = get_barycenter(df,'y')
    _temp.baryZ = get_barycenter(df,'z')

    _temp.Etotal = total_energy(df)
    _temp.dR = get_dr(_temp.baryX,_temp.baryY)
    _temp.hits = [float(hit) for hit in df.groupby(['event']).size().values]

    # E1-E20
    _temp.E1to20 = enrgy_bw_layer(df,1,20)
    # E21-E28
    _temp.E21to28 = enrgy_bw_layer(df,21,28)
    #E1-E28
    _temp.E1to28 = enrgy_bw_layer(df,1,28)
    #E29-E40
    _temp.E29to40 = enrgy_bw_layer(df,29,40)
    # E1-E8
    # _temp.E1to8 = enrgy_bw_layer(df,1,8)

    # E1-E2 / Etot
    _temp.E1to2byEtot = enrgy_bw_layer(df,1,2)
    # E26-E28 / Etot
    _temp.E26to28byEtot = enrgy_bw_layer(df,26,28)
    # E1-E10 / Etot
    _temp.E1to10byEtot = enrgy_bw_layer(df,1,10)

    # E1-E8 / E1-E20
    # _temp.E1to8byE1to20 = _temp.E1to8 / _temp.E1to20
    # E1-E20 / E21-E28
    _temp.E1to20byE21to28 = _temp.E1to20 / _temp.E21to28

    # get per layer energy
    _temp = pd.concat([_temp,get_layers_df(df,_temp.index.unique(),layers)],axis=1)
    dataset = dataset.append(_temp)
    df, _temp = None, None

dataset.index.name = 'event'
dataset.beamE = beamE

print("Writing processed data into HDF5 file")
hdf = pd.HDFStore(output_file, mode = 'a') # open file in append mode
hdf.put('dataset',dataset,data_columns=True)
hdf.close()
dataset = None
print("Process complete.")
