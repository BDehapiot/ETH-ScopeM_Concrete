#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Function: norm_gcn --------------------------------------------------------

def norm_gcn(arr, sample_fraction=1, mask=None):
    
    """ 
    Global contrast normalization.

    Array is normalized by substracting the mean and dividing by the 
    standard deviation. Calculation can be restricted to a random fraction 
    (sample_fraction) and/or a given selection (mask). NaNs are ignored.
    
    Parameters
    ----------
    arr : ndarray (uint8, uint16 or float)
        Array to be normalized.
        
    sample_fraction : float
        Fraction of the array to be considered for mean and standard deviation
        calculation. Must be between 0 and 1.
        
    mask : ndarray (bool)
        Selection of the array to be considered for mean and standard deviation
        calculation. Mask and array must be of the same shape.
                
    Returns
    -------  
    arr : ndarray (float)
        Normalized array
    
    """
    
    # Check inputs
    if arr.dtype != "float32":
        arr = arr.astype("float32")
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("sample_fraction should be float between 0 and 1")
    if mask is not None and mask.shape != arr.shape:
        raise ValueError("array and mask should have the same shape")
    
    # Extract values
    val = arr.ravel()
    if mask is not None:
        val = val[mask.ravel()]
    if sample_fraction < 1:
        val = np.random.choice(val, size=int(arr.size * sample_fraction))
    val = val[~np.isnan(val)]
        
    # Normalize
    arr -= np.mean(val)
    arr /= np.std(val) 
    
    return arr

#%% Function: norm_pct --------------------------------------------------------

def norm_pct(
        arr,
        pct_low=0.01,
        pct_high=99.99,
        sample_fraction=1,
        mask=None
        ):
    
    """ 
    Percentile normalization.

    Array is normalized from 0 to 1 considering a range determined by a low and
    a high percentile value (pct_low and pct_high). Out of range values are 
    clipped and NaNs are ignored. Calculation can be restricted to a random 
    fraction (sample_fraction) and/or a given selection (mask).

    Parameters
    ----------
    arr : ndarray (uint8, uint16 or float)
        Array to be normalized.
        
    pct_low : float
        Percentile to determine the low value of the normalization range.
        pct_low must be >= 0 and < pct_high. If pct_low == 0, low value is 
        equal to the minimum value of the array. 

    pct_high : float
        Percentile to determine the high value of the normalization range.
        pct_high must be > pct_low and <= 100. If pct_high == 100, high value 
        is equal to the maximum value of the array.
        
    sample_fraction : float
        Fraction of the array to be considered for mean and standard deviation
        calculation. Must be between 0 and 1.
        
    mask : ndarray (bool)
        Selection of the array to be considered for mean and standard deviation
        calculation. Mask and array must be of the same shape.
                
    Returns
    -------  
    arr : ndarray (float)
        Normalized array
    
    """
    
    # Check inputs
    if arr.dtype != "float32":
        arr = arr.astype("float32")
    if pct_low < 0 or pct_low >= pct_high:
        raise ValueError("pct_low should be >= 0 and < pct_high")
    if pct_high > 100 or pct_high <= pct_low:
        raise ValueError("pct_high should be <= 100 and > pct_low")
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("sample_fraction should be float between 0 and 1")
    if mask is not None and mask.shape != arr.shape:
        raise ValueError("array and mask should have the same shape")
        
    # Extract values
    val = arr.ravel()
    if mask is not None:
        val = val[mask.ravel()]
    if sample_fraction < 1:
        val = np.random.choice(val, size=int(arr.size * sample_fraction))
    val = val[~np.isnan(val)]
    
    # Normalize
    if pct_low == 0: pLow = np.nanmin(arr)
    else: pLow = np.percentile(val, pct_low)
    if pct_high == 100: pHigh = np.nanmax(arr)
    else: pHigh = np.percentile(val, pct_high)
    np.clip(arr, pLow, pHigh, out=arr)
    arr -= pLow
    arr /= (pHigh - pLow)
        
    return arr