import numpy as np
from scipy.stats import binned_statistic


def get_barycenter(df,axis,chamber=None):
    '''
    Calulates all three barycenters, valid dimension are x,y,z.
    there are two chamber in HGCal hadronic and electromagnatic.
    '''

    _timesE = axis+'_timesE' # (x/y/z)_timesE
    df_group = df.groupby('event')
    totE = df_group.rechit_energy.sum() # total sum of energy per event
    
    if not chamber:
        barycenter = df_group[_timesE].sum()/totE # (energy * location) / (total sum of energy per event)
    
    # assuming layers setup is 40 : 1-28[EE] + 29-40[FH]
    elif(chamber == 'EE'): # Electromagnatic chamber
        df_sel = df[df.rechit_layer < 29].copy()
        df_group = df_sel.groupby('event')
        
        # taking energy deposite only in layer 1-28
        # comment if need to consider total energy
        totE = df_group.rechit_energy.sum()
        barycenter = df_group[_timesE].sum()/totE
    
    elif(chamber == 'FH'): # Hadronic chamber
        df_sel = df[(df.rechit_layer > 28) & (df.rechit_layer < 40)].copy()
        df_group = df_sel.groupby('event')
        
        # taking energy deposite only in layer 29-40
        # comment if need to consider total energy
        # totE = df_group.rechit_energy.sum()
        barycenter = df_group[_timeE].sum()/totE
    else:
        raise Exception("Invalid Chamber provided, EE and FH are the only valid chambers or select None")

    return barycenter


def get_dr(bary_x, bary_y):

    # dR = sqrt(dx^2 + dy^2)
    dr = np.hypot(bary_x, bary_y) 
    return dr

def get_longitudinal_profile(df):
    
    layer_sum = df.groupby(['event','rechit_layer'])['rechit_energy'].sum()
    
    mdn_layer_sum = layer_sum.reset_index().groupby('rechit_layer')['rechit_energy'].median()
    avg_layer_sum = layer_sum.reset_index().groupby('rechit_layer')['rechit_energy'].mean()
    
    return avg_layer_sum, mdn_layer_sum


def get_radial_profile(df,chamber=None):
    ''' 
    Radial spread profile.
    It returns layer and dr (= rechit_r - bary_r) per layer.
    These two quantities are evaluated as the median over all the events.
    bary_x and bary_y should be according to chamber, do not pass bary_x of all layers
    to get_radial_profile of EE.
    '''  
    
    _df = df.copy()
    
    # if chamber is EE consider only 1-28 layers to get radial profile
    bins = 40
    if chamber == 'EE':
        _df = df[df.rechit_layer < 29].dropna()
        bins = 28
        
    df_group = _df.groupby('event') # groupby events
    totE = df_group.rechit_energy.sum()
    
    dx = _df.rechit_x - df_group.x_timesE.sum()/totE # baryX
    dy = _df.rechit_y - df_group.y_timesE.sum()/totE # baryY
    
    dr = np.hypot(dx,dy)
    
    x = _df.rechit_layer
    
    x_med, y_med = binned_statistic(x, [x,dr], bins = 40, statistic='median').statistic
    
    sel = ~np.isnan(x_med)
    y_med = y_med[sel]
    x_med = x_med[sel]

    return x_med, y_med


def avg_hits_per_layer(df):

    df_nhits = df.groupby(['event','rechit_layer']).size().reset_index() # no of hits per event per layer
    df_nhits.columns = ['event','layer','nhits']
    
    # avarage no of hits per layer all events
    avg_hits = df_nhits.groupby('layer')['nhits'].mean()
    
    return avg_hits

def enrgy_bw_layer(df,layer1,layer2):
    sel = ( df.rechit_layer > layer1-1) & ( df.rechit_layer < layer2+1 )
    return df[sel].groupby(['event']).rechit_energy.sum()

def total_energy(df,chamber=None):
    if chamber == 'EE':
        return df[df.rechit_layer < 29].groupby('event').rechit_energy.sum()
    else:
        return df.groupby('event').rechit_energy.sum()

def get_hypot(df):
    df_group = df.groupby('event') # groupby events
    totE = df_group.rechit_energy.sum()
    dx = df.rechit_x - df_group.x_timesE.sum()/totE # baryX
    dy = df.rechit_y - df_group.y_timesE.sum()/totE # baryY
    return np.hypot(dx,dy)