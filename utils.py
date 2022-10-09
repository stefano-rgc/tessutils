# Built-in modules
import re
# External modules
import numpy as np
from astropy.io import fits
from astropy.time import Time

def tpf_name_pattern():
    """Default name pattern for TPFs"""
    return 'tic{TIC}_sec{SECTOR}.fits'

def contains_TIC_and_sector(pattern):
    """Ensure pattern containing keywords for TIC and sector"""
    try:
        if '{TIC}' in pattern and '{SECTOR}' in pattern:
            pass
    except Exception as e:
        raise ValueError('Pattern must contain keywords {TIC} and {SECTOR}.')

def contains_TIC(pattern):
    """Ensure pattern containing keyword for TIC"""
    try:
        if '{TIC}' in pattern:
            pass
    except Exception as e:
        raise ValueError('Pattern must contain keyword {TIC}.')

def contains_two_numbers(name):
    """Ensure that `name` contains two distinct numbers"""
    if not re.match('.*?(\d+)[^0-9]+(\d+).*?',name):
        raise ValueError('`name` must contain two numbers separated by a non-number.')

def contain_one_number(name):
    """Ensure that `name` contains one number"""
    if not re.match('.*?(\d+).*?',name):
        raise ValueError('`name` must contain one number.')

def return_TIC_and_sector(name, pattern=None):
    """Return TIC and sector numbers from a str by matching `name` with
    `pattern`. The latter must contain the keywords {TIC} and {SECTOR}"""
    contains_two_numbers(name)
    contains_TIC_and_sector(pattern)
    # Substitute the keywords for regular expressions
    _pattern = pattern.format(TIC='(\d+)', SECTOR='\d+')
    if (match := re.match(_pattern,name)):
        TIC = int(match.group(1))
    else:
        raise ValueError(f'`name` ({name}) does not match `pattern` ({pattern}).')
    # Substitute the keywords for regular expressions
    _pattern = pattern.format(TIC='\d+', SECTOR='(\d+)')
    if (match := re.match(_pattern,name)):
        sector = int(match.group(1))
    else:
        raise ValueError(f'`name` ({name}) does not match `pattern` ({pattern}).')
    return TIC, sector 

def return_sector(*args, **kwargs):
    """Return sector number from str containing keywords {TIC} and {SECTOR}"""
    return return_TIC_and_sector(*args, **kwargs)[1]

def return_TIC_1(*args, **kwargs):
    """Return TIC number from str containing keywords {TIC} and {SECTOR}"""
    return return_TIC_and_sector(*args, **kwargs)[0]

def return_TIC_2(name, pattern=None):
    """Return TIC numbers from a str by matching `name` with
    `pattern`. The latter must contain the keyword {TIC}"""
    contain_one_number(name)
    contains_TIC(pattern)
    # Substitute the keywords for regular expressions
    _pattern = pattern.format(TIC='(\d+)')
    if (match := re.match(_pattern,name)):
        TIC = int(match.group(1))
    else:
        raise ValueError(f'`name` ({name}) does not match `pattern` ({pattern}).')
    return TIC

def chunks(lst, n):
    """
    Source
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    
    Purpose
        Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def minmax(x):
    """Return min and max of array `x`"""
    if isinstance(x, Time):
        return x.min(), x.max()
    return np.min(x), np.max(x)

def print_err(err_msg, prepend=''):
    '''Convenience function to print error messages'''
    err_msg = prepend + err_msg
    print(err_msg)
    return err_msg

def get_header_info(fitsFile):
    "Return a list containing the header information of each header unit."
    # Save header information from original FITS file
    hdulist = []
    ext = 0
    while True:
        try:
            hdulist.append( fits.getheader(fitsFile, ext=ext) )
            ext += 1
        # No more extentions in header
        except IndexError:
            break
        # Unknown error
        except Exception as e:
            e_name = e.__class__.__name__
            print(f'Unexpected exception when reading headers from FITS file. Exception: -> {e_name}: {e}.')
            break
    return hdulist

def parse_lc_time_units(lc, short=False):
    """Parse the light curve object to decide on the plotted units.
    Args:
        lc (lightkurve.lightcurve.LightCurve object): Light curve
    Source:
        https://github.com/lightkurve/lightkurve/blob/9763e49b9ae7e794685fed7e0043b5cd67779564/src/lightkurve/lightcurve.py#L1880-L1889
    Returns:
        ylabel (str)
    """
    # Default xlabel
    if not hasattr(lc.time, "format"):
        label = "Phase"
    elif lc.time.format == "bkjd":
        if not short:
            label = "Time - 2454833 [BKJD days]"
        else:
            label = "[BKJD days]"
    elif lc.time.format == "btjd":
        if not short:
            label = "Time - 2457000 [BTJD days]"
        else:
            label = "[BTJD days]"
    elif lc.time.format == "jd":
        label = "Time [JD]"
    else:
        label = "Time"
    return label