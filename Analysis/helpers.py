
import numpy as np
import pandas as pd

def get_layers_df(df,index,layers):
        
    _temp = df.groupby(['event','rechit_layer']).rechit_energy.sum().groupby('event').apply(list)
    
    _list = np.full([len(_temp),len(layers)],np.nan)
    count = 0
    for e in _temp:
        _list[count][:len(e)] = e
        count += 1
    _list =  pd.DataFrame(_list,columns = layers,index = index)
    _list.index.name = 'event'
    return _list