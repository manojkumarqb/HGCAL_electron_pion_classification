
import uproot
import concurrent.futures
import uproot._connect._pandas


def load_data(file_name,columns,key='rechitntupler/hits',entrystart=None, entrystop=None, flatten=True, threads=16):
    '''
    This function loads experiment/simulation data into system memory as pandas DataFrames,
    'the data should not be large'. choose entrystart(starting point of event) and entrystop
    (ending point of event) accordingly.
    entrystop - entrystart = no of event's data loaded into memory

    To not flatten dataset make flatten = False, this will save memory
    There is no need to load dataset into cache.
    variables event_id and rechit_layer are not needed, rechit_layer = rechit_z and event_id is insignificant.
    '''

    '''
    No need to fetch event id, the event will be the index and if some of the events are missing
    this will allot event nos in sequence.
    '''
    
    file_content = uproot.open(file_name) # pointer to dataset file, loading metadata
                                            #(eg.- root directories, branches keys etc.)
    hits = file_content[key] # hits corresponding to the key
        
    executor = concurrent.futures.ThreadPoolExecutor(threads) # threads in pool
    
    df = hits.pandas.df(columns, entrystart = entrystart,entrystop = entrystop,
                        flatten = flatten,executor = executor)
    df = df.reset_index(level=1,drop=True)
    df.index.names = ['event']
    return df


# def load_chunks():
#     ntuple_dir, ntuple_name, branches, mask="", chunksize=None, key="rechitntupler/hits", isSim=False


def x_times_e(df):
    '''
    Generating extra features from exsting feature
    (x/y/z)_timesE = rechit_(x/y/z) * rechit_energy
    '''
    for x in ['x','y','z']:
        _temp = x + '_timesE'
        df[_temp] = df.rechit_energy * df['rechit_'+x]