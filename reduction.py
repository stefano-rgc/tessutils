# Built-in modules
import signal, warnings, pickle, re
from time import sleep
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from collections import deque
from multiprocessing import Pool
# External modules
import numpy as np
import pandas as pd
import lightkurve as lk 
from scipy import ndimage
from astropy.stats.funcs import median_absolute_deviation as MAD
import astropy.units as u
from astropy.modeling import fitting, functional_models
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astropy.time import Time
from joblib import Parallel, delayed
from tqdm import tqdm
# Local modules
from . import utils

def download_tpf(TIC,
                 imsize=20,
                 pattern='tic{TIC}_sec{SECTOR}.fits',
                 outputdir='./tpfs',
                 max_queryTries=3,
                 max_downloadTries=10,
                 max_saveTries=2,
                 sectors=None,
                 overwrite=False,
                 progressbar=False,
                 ncores=None):
    """
    Purpose:
        Download a TESS cut image(s) or target pixel file(s) (TPF) for (a) given 
        TIC number(s).

    Args:
        TIC (int | list[int] | numpy.ndarray | pandas.core.series.Series):
            TIC number(s) to download.
        imsize (int, optional):
            Size in pixels of the square image. Defaults to 20.
        pattern (str, optional):
            Pattern name to save the TPF as a FITS file. Pattern must contain
            the following two raw string used as key words: {TIC} and {SECTOR}.
            Defaults to 'tic{TIC}_sec{SECTOR}.fits'.
        outputdir (str, optional):
            Directory used to store the FITS files. Defaults to ./tpfs.
        max_queryTries (int, optional):
            Maximum numer of attempts to query the MAST database for the TIC
            star. Defaults to 3.
        max_downloadTries (int, optional):
            Maximum numer of attempts to download data from the MAST database.
            Defaults to 10.
        max_saveTries (int, optional):
            Maximum numer of attempts to save the FITS files. Defaults to 2.
        sectors (None | int | list[int], optional):
            TESS sectors to download. Defaults to None donwloads all.
        overwrite (bool, optional):
            Overwrite FITS files if already downloaded. Defaults to False.
        progressbar (bool, optional):
            Show a progress bar. Defaults to False.
        ncores (int, optional):
            Number of parallel processes to download a list of TIC numbers. All
            sectors available for a TIC number are under one process, i.e., the
            parallelization happens at TIC level and not at sector level.
            Defaults to None max out available cores.

    Returns:
        None
        
    Examples:
        1. Download all TPFs available for TIC 374944608:
            > TIC = 374944608
            > download_tpf(TIC)
        2. Download only TESS sectors 1 and 2 for TIC 374944608 and 38845463.
        Parallelize the download using 2 cores and display a progress bar:
            > TICs = [38845463,374944608]
            > download_tpf(TICs, progressbar=True, ncores=2, sectors=[1,2])
    """    
    
    # Handle a list-like input
    valid_types = (list,np.ndarray,pd.core.series.Series)
    if isinstance(TIC,valid_types):
        # Update kw arguments of the function
        _download_tpf = partial(download_tpf, imsize=imsize,
                                              pattern=pattern,
                                              outputdir=outputdir,
                                              max_queryTries=max_queryTries,
                                              max_downloadTries=max_downloadTries,
                                              max_saveTries=max_saveTries,
                                              sectors=sectors,
                                              overwrite=overwrite,
                                              progressbar=False,
                                              ncores=1)
        with Pool(ncores) as pool:
            it = pool.imap_unordered(_download_tpf, TIC)
            if progressbar:
                it = tqdm(it, total=len(TIC))
            # Exhaust the iterator
            deque(it, maxlen=0)
        return   
    
    utils.contains_TIC_and_sector(pattern)
    outputdir = Path('tpfs') if outputdir is None else Path(outputdir)
    if not outputdir.exists():
        outputdir.mkdir(parents=True)
    
    # Search MAST for all FFIs available for TIC
    tries = 1
    while True:
        if tries > max_queryTries:
            print(f'Skipped TIC={TIC}: Maximum number of MAST query retries ({max_queryTries}) exceeded.')
            return
        try: 
            tesscuts = lk.search_tesscut(f'TIC {TIC}')
            break # Exit the loop if TIC is found
        except Exception as e:
            e_name = e.__class__.__name__
            print(f'MAST query attempt {tries}, TIC = {TIC}. Excepion -> {e_name}: {e}')
        tries += 1

    if len(tesscuts) == 0:
        print(f'No images found for TIC={TIC}.')
        return

    # Check that there is only one returned ID
    ids = np.unique(tesscuts.table['targetid'].data)
    if not ids.size == 1:
        print(f'The MAST query returned multiple ids: {ids}')
        print('No FITS files saved')
        return
    # Check that the returned ID matches the TIC number
    if str(TIC) != re.match('TIC (\d+)',ids.item()).group(1):
        print(f'The MAST query returned a different id: {ids}')
        print('No FITS files saved')
        return
    
    # Get sector numbers
    try:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    except KeyError:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['mission'] ])
    # Filter only requested sectors
    if not sectors is None:
        if isinstance(sectors, int):
            sectors = [sectors]
        ind =[True if sec in sectors else False for sec in secs.astype('int32')]
        tesscuts = tesscuts[ind]
    # Get sector numbers again
    try:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    except KeyError:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['mission'] ])
    secs = secs.astype('int32')
    
    # Generate output names
    outputnames = np.array([outputdir/Path(pattern.format(TIC=TIC, SECTOR=sec)) for sec in secs])
    
    if not overwrite:
        # Skip already downloaded files
        files = np.array([file.exists() for file in outputnames])
        ind = np.argwhere(files==True).flatten()
        if len(ind) > 0:
            skipped_secs = ','.join(secs[ind].astype(str))
            print(f'Skipped: Already downloaded sectors for TIC={TIC}: {skipped_secs}.')
            ind = np.argwhere(files==False).flatten().tolist()
            tesscuts = tesscuts[ind]
            if len(tesscuts) == 0:
                print(f'Skipped: No new images to download for TIC={TIC}.')
                return

    # Download TESS cut or target pixel file
    tries = 1
    while True:
        if tries > max_downloadTries:
            print(f'Skipped TIC={TIC}: Maximum number of download retries ({max_downloadTries}) exceeded.')
            return
        try:
            tpfs = tesscuts.download_all(cutout_size=imsize) # TODO: This may be a chance to use an async funtion or method
            break # Exit the loop if download is successful
        except TypeError as e:
            e_name = e.__class__.__name__
            print(f'Skipped TIC={TIC}: There seems to be a problem with the requested image. Excepion -> {e_name}: {e}.')
            return
        except Exception as e:
            # If exception rised
            e_name = e.__class__.__name__
            if e_name == 'SearchError':
                print(f'Skipped TIC = {TIC}: There seems to be a problem with the requested image. Excepion -> {e_name}: {e}.')
                return
            print(f'Download try number {tries} for TIC={TIC}. Excepion -> {e_name}: {e}')
            # ? Need to add return statement here ?
        tries += 1

    # Save as FITS files
    for tpf in tpfs:
        # Store TIC number in the header
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tpf.header.set('TICID',value=TIC)
        sector = tpf.sector 
        outputname = outputdir/Path(pattern.format(TIC=TIC, SECTOR=sector))
        tries = 1
        # Attempt to write FITS file
        while True:
            if tries > max_saveTries:
                print(f'Skipped TIC={TIC}: Maximum number of retries ({max_saveTries}) exceeded.')
                return
            try:
                tpf.to_fits(outputname.as_posix(), overwrite=overwrite)
                break # Exit the loop if save is successful
            except OSError as e:
                print('When saving FITS file for TIC={TIC}. Excepion -> OSError: {e}.')
            except Exception as e:
                e_name = e.__class__.__name__
                print(f'Attempt {tries} when saving FITS file, TIC = {TIC}. Excepion -> {e_name}: {e}.')
                sleep(0.5) # Allow time before next attempt
            tries += 1

        # Message for successful save
        print(f'Saved: {outputname.as_posix()}')
    
def normalize_lightCurve(lc):
    '''Function applied to light curves of individual TESS sectors before stitching them'''
    median = np.median(lc.flux)
    return (lc-median)/median

def threshold_mask(image,
                   threshold=3,
                   reference_pixel='center'):
    """
    Returns an aperture mask creating using the thresholding method.
    This method will identify the pixels in the TargetPixelFile which show
    a median flux that is brighter than `threshold` times the standard
    deviation above the overall median. The standard deviation is estimated
    in a robust way by multiplying the Median Absolute Deviation (MAD)
    with 1.4826.
    If the thresholding method yields multiple contiguous regions, then
    only the region closest to the (col, row) coordinate specified by
    `reference_pixel` is returned.  For exmaple, `reference_pixel=(0, 0)`
    will pick the region closest to the bottom left corner.
    By default, the region closest to the center of the mask will be
    returned. If `reference_pixel=None` then all regions will be returned.

    Source
    ----------
    LightKurve module
    
    Notes
    ----------
    The original only works as a method. I (Stefano) made it a function by
    adapting the necessary lines. Such a lines are indicated with a comment
    that says Stefano.  

    Parameters
    ----------
    threshold : float
        A value for the number of sigma by which a pixel needs to be
        brighter than the median flux to be included in the aperture mask.
    reference_pixel: (int, int) tuple, 'center', or None
        (col, row) pixel coordinate closest to the desired region.
        For example, use `reference_pixel=(0,0)` to select the region
        closest to the bottom left corner of the target pixel file.
        If 'center' (default) then the region closest to the center pixel
        will be selected. If `None` then all regions will be selected.

    Returns
    -------
    aperture_mask : ndarray
        2D boolean numpy array containing `True` for pixels above the
        threshold.
    """
    if reference_pixel == 'center':
        # reference_pixel = (image.shape[2] / 2, image.shape[1] / 2) # Original line
        reference_pixel = (image.shape[1] / 2, image.shape[0] / 2) # Modified by Stefano
    vals = image[np.isfinite(image)].flatten()
    # Calculate the theshold value in flux units
    mad_cut = (1.4826 * MAD(vals) * threshold) + np.nanmedian(image)
    # Create a mask containing the pixels above the threshold flux
    threshold_mask = np.nan_to_num(image) > mad_cut
    if (reference_pixel is None) or (not threshold_mask.any()):
        # return all regions above threshold
        return threshold_mask
    else:
        # Return only the contiguous region closest to `region`.
        # First, label all the regions:
        labels = ndimage.label(threshold_mask)[0]
        # For all pixels above threshold, compute distance to reference pixel:
        label_args = np.argwhere(labels > 0)
        distances = [np.hypot(crd[0], crd[1])
                     for crd in label_args - np.array([reference_pixel[1], reference_pixel[0]])]
        # Which label corresponds to the closest pixel?
        closest_arg = label_args[np.argmin(distances)]
        closest_label = labels[closest_arg[0], closest_arg[1]]
        return labels == closest_label

def query_TIC(target,
              target_coord,
              tic_id=None,
              search_radius=600.*u.arcsec,
              **kwargs):
        """
        Retrieving information from the TESS input catalog. 

        Source
        ----------
        Courtesy of Dr. Timothy Van Reeth

        Notes
        ----------
        I (Stefano) modified the behaviour when `tic_id` is given.


        Parameters
        ----------
        target:
            Target name
        target_coord (optional):
            Target coordinates (astropy Skycoord)
        search_radius:
            TIC entries around the target coordinaes wihtin this radius are considered.
        **kwargs:
            dict to be passed to astroquery.Catalogs.query_object or query_region.
        """
        
        def _tic_handler(lc,signum):
            '''Supporting function of `query_TIC`'''
            print('the query of the TIC is taking a long time... Something may be wrong with the database right now...')

        deg_radius = float(search_radius / u.deg)
        arc_radius = float(search_radius / u.arcsec)
        
        tic = None
        tess_coord = None
        tmag = None 
        nb_coords = []
        nb_tmags = []
        tic_index = -1
        
        try:
            # The TIC query should finish relatively fast, but has sometimes taken (a lot!) longer.
            # Setting a timer to warn the user if this is the case...
            signal.signal(signal.SIGALRM,_tic_handler)
            signal.alarm(30) # This should be finished after 30 seconds, but it may take longer...
            
            catalogTIC = Catalogs.query_region(target_coord, catalog="TIC", radius=deg_radius,**kwargs)
            signal.alarm(0)
            
        except:
            print(f"no entry could be retrieved from the TIC around {target}.")
            catalogTIC = []
        
        if(len(catalogTIC) == 0):
            print(f"no entry around {target} was found in the TIC within a {deg_radius:5.3f} degree radius.")
        
        else:
            if not (tic_id is None):
                # tic_index = np.argmin((np.array(catalogTIC['ID'],dtype=int) - int(tic_id))**2.) # Original line
                tic_index = np.argwhere(catalogTIC['ID'] == str(tic_id)) # Modified by Stefano
                if tic_index.size == 0:
                    return '-1', None, None, None, None

                else:
                    tic_index = tic_index.item()
            else:
                tic_index = np.argmin(catalogTIC['dstArcSec'])
        
            if(tic_index < 0):
                print(f"the attempt to retrieve target {target} from the TIC failed.")
            
            else:
                tic = int(catalogTIC[tic_index]['ID'])
                ra = catalogTIC[tic_index]['ra']
                dec = catalogTIC[tic_index]['dec']
                tmag = catalogTIC[tic_index]['Tmag']
                
                # Retrieve the coordinates
                tess_coord = SkyCoord(ra, dec, unit = "deg")
                
                # Collecting the neighbours
                if(len(catalogTIC) > 1):
                    for itic, tic_entry in enumerate(catalogTIC):
                        if(itic != tic_index):
                            nb_coords.append(SkyCoord(tic_entry['ra'], tic_entry['dec'], unit = "deg"))
                            nb_tmags.append(tic_entry['Tmag'])
        
        nb_tmags = np.array(nb_tmags)
        
        return tic, tess_coord, tmag, nb_coords, nb_tmags

def check_aperture_mask(aperture,
                        aperture_mask_max_elongation=14,
                        aperture_mask_min_pixels=4,
                        prepend_err_msg=''):
    """
    Purpose:
        Check if aperture satisfies geometric criteria, ie, is not too elongated
        and not too small.

    Args:
        aperture (numpy.ndarray):
            Aperture mask.
        aperture_mask_max_elongation (int, optional):
            Triggers when a 2D aperture mask has 4 or less columns (rows) and at
            least one row (column) larger that `aperture_mask_max_elongation` pixels. In such
            a case, the function returns `OK_aperture=Flase`. Default is 14.
        aperture_mask_min_pixels (int, optional):
            Minimum number of pixels in the aperture so that the function
            returns `OK_aperture=True`. Default is 4.
        prepend_err_msg (str, optional):
            Sting to prepend to the error message. Default is ''.

    Raises:
        ValueError 
        ValueError
        AttributeError

    Returns:
        OK_aperture (bool):
            True if aperture satisfies geometric criteria, False otherwise.
        err_msg (str):
            Error message.
    """
    # Initializations
    err_msg = ''
    OK_aperture = True
    # Convert from boolean to int
    aperture = aperture.astype(int)
    # If not apperture found
    if not np.any(aperture) and OK_aperture:
        err_msg = utils.print_err('Not aperture found.', prepend=prepend_err_msg)
        OK_aperture = False
    # If too elongated aperture (column)
    if not np.all(aperture.sum(axis=0) <= aperture_mask_max_elongation) and OK_aperture:
        # Check if is a bad defined aperture
        if len( set( aperture.sum(axis=0) ) ) < 4:
            err_msg = utils.print_err('Too elongated aperture (column).', prepend=prepend_err_msg)
            OK_aperture = False
        if np.any(aperture.sum(axis=0) == 0):
            err_msg = utils.print_err('Too elongated aperture (column), bad TESS image.', prepend=prepend_err_msg)
            OK_aperture = False
    # If too elongated aperture (row)
    if not np.all(aperture.sum(axis=1) <= aperture_mask_max_elongation) and OK_aperture:
        # Check if is a bad defined aperture
        if len( set( aperture.astype(int).sum(axis=1) ) ) < 4:
            err_msg = utils.print_err('Too elongated aperture (row).', prepend=prepend_err_msg)
            OK_aperture = False
        if np.any(aperture.sum(axis=1) == 0):
            err_msg = utils.print_err('Too elongated aperture (row), bad TESS image.', prepend=prepend_err_msg)
            OK_aperture = False
    # If too small aperture
    if not np.sum(aperture) >= aperture_mask_min_pixels and OK_aperture:
        err_msg = utils.print_err('Too small aperture.', prepend=prepend_err_msg)
        OK_aperture = False
    return OK_aperture, err_msg

def find_fainter_adjacent_pixels(seeds,
                                 image,
                                 max_iter=100):    
    """
    Purpose:
        Given an initial pixel(s) a.k.a. seed, find surrounding pixels until the
        pixel value increases.

    Args:
        seeds (numpy.ndarray):
            Array of initial pixels.
        image (numpy.ndarray):
            Image to search for fainter pixels.
        max_iter (int, optional):
            Maximun number of iterations. Defaults to 100.

    Raises:
        ValueError
        ValueError
        AttributeError

    Returns:
        mask (numpy.ndarray):
            A boolean mask where TRUE indicates pixels with decreasing value
            around the seed.
    """
    # Check that the dimension of `seeds` is correct
    error_msg = '`seeds` has to be a 2D Numpy array whose second dimension has lengh 2. E.g.: np.array([[0,1], [5,5], [7,8]])'
    try:
        if seeds.ndim != 2:
            raise ValueError(error_msg)
        if seeds.shape[1] != 2:
            raise ValueError(error_msg)
    except AttributeError:
        raise AttributeError(error_msg)
    # Here we'll keep track of which pixels are what:
    # * -1: Not part of the mask.
    # *  0: Part of the mask and previous seed
    # * +1: Part of the mask
    score = np.repeat(-1,image.size).reshape(image.shape)
    # The center defined by the initial seeds
    score[seeds[:,0],seeds[:,1]] = 1
    # Initialize the counter
    counter = 0
    while True:
        # Check the counter
        if counter > max_iter:
            print(f'Maximum number of iterations exceeded: max_iter={max_iter}')
            break
        # Find which pixels use as centers
        centers = np.argwhere(score==1)
        # List to store the indices of the pixels to be included as part of the mask
        inds = []
        # Evaluate the condition for each center (i.e., search for adjacent fainter pixels)
        for center in centers:
            # Find the 4 adjacent pixels of pixel center
            mask = np.repeat(False,image.size).reshape(image.shape)
            mask[center[0],center[1]] = True
            mask = ~ndimage.binary_dilation(mask)
            mask[center[0],center[1]] = True
            masked_image = np.ma.masked_where(mask, image)
            # Find which of the adjacent pixels are not brighter than pixel center
            try:
                ind = np.argwhere(masked_image <= masked_image.data[center[0],center[1]])
            except u.UnitConversionError:
                ind = np.argwhere(masked_image <= masked_image.data.value[center[0],center[1]])
            inds.append(ind)
        # If any pixel fainter than the center one
        if len(inds) > 0:
            ind = np.concatenate(inds)
            # Extend the mask
            score[ind[:,0],ind[:,1]] = 1
            # Flag the all previous centers
            score[centers[:,0],centers[:,1]] = 0
        # If no pixel fainter than the center one
        else:
            break
    mask = score + 1
    mask = mask.astype(bool)
    return mask

def mag2flux(mag, zp=20.60654144):
    """
    Convert from magnitude to flux using scaling relation from
    aperture photometry. This is an estimate.

    The scaling is based on fast-track TESS data from sectors 1 and 2.

    Source
    ----------
    Source: https://tasoc.dk/code/_modules/photometry/utilities.html#mag2flux
    
    Parameters:
        mag (float): Magnitude in TESS band.

    Returns:
        float: Corresponding flux value
    """
    return np.clip(10**(-0.4*(mag - zp)), 0, None)

def contamination(info,
                  prepend_err_msg='',
                  max_num_of_neighbour_stars=40):
    """
    Purpose:
        Calculate the fraction of flux in the aperture the mask that comes from
        neighbour stars. Done by means of fitting 2D Gaussians and a plane to
        the image.
        
    Args:
        info:
            SimpleNamespace object with the following attributes:
                * info.median_image
                * info.masks.aperture
                * info.masks.background
                * info.target.pix
                * info.target.mag
                * info.neighbours_used.pix
                * info.neighbours_used.mag
            to which will be added:
                * info.fit
        prepend_err_msg (str, optional):
            String to prepend to the error message. Defaults to ''.
        max_num_of_neighbour_stars (int, optional):
            Maximum number of neighbour stars to fit. If there are more
            neighbour stars, then they are truncated.
    Returns:
        fitted_image (numpy.ndarray):
            Image with the fitted 2D Gaussians and plane.
        err_msg (str):
            Error message.
    """
    # Unpack some information
    image = info.median_image
    mask_aperture = info.masks.aperture
    mask_background = info.masks.background
    target_coord_pixel = info.target.pix
    target_tmag = info.target.mag
    nb_coords_pixel = info.neighbours_used.pix
    nb_tmags = info.neighbours_used.mag
    # Initializations
    err_msg = ''
    # Set a maximum of 40 neighbour stars to fit
    if nb_tmags.size > 0:
        nb_tmags = nb_tmags[:max_num_of_neighbour_stars]
        nb_coords_pixel = nb_coords_pixel[:max_num_of_neighbour_stars,:]
    # Gaussian locations
    locations = np.array([*target_coord_pixel,*nb_coords_pixel])
    xpos = locations[:,0]
    ypos = locations[:,1]
    # Convert the magnitudes of the stars to flux
    fluxes = mag2flux( np.concatenate( [np.array([target_tmag]), nb_tmags] ) )
    # Create model
    Gaussian2D = functional_models.Gaussian2D
    Planar2D = functional_models.Planar2D
    Gaussians = [ Gaussian2D(amplitude=a,
                             x_mean=x, 
                             y_mean=y, 
                             x_stddev=1, 
                             y_stddev=1) for a,x,y in zip(fluxes,xpos,ypos) ]
    nGaussians = len(Gaussians)
    model = np.sum(Gaussians)
    def tie_sigma(model):
        '''Used to constrain the sigma of the Gaussians when performing the fit in `contamination()`'''
        try:
            return model.x_stddev
        except AttributeError:
            return model.x_stddev_0
    def tie_amplitude(model,factor=1):
        '''Used to constrain the amplitude of the Gaussians when performing the fit in `contamination()`'''
        return model.amplitude_0*factor
    # Set the constrains in the model
    if nGaussians == 1:
        getattr(model,'x_mean').fixed = True
        getattr(model,'y_mean').fixed = True
        getattr(model,'y_stddev').tied = tie_sigma
        getattr(model,'amplitude').min = 0.0
    else:
        for i in range(nGaussians):
            # Tie all Gaussian Sigmas to the x-dimension Sigma of the target star
            getattr(model,f'y_stddev_{i}').tied = tie_sigma
            getattr(model,f'x_stddev_{i}').tied = tie_sigma
            # Fix the Gaussian positions
            getattr(model,f'x_mean_{i}').fixed = True
            getattr(model,f'y_mean_{i}').fixed = True
            # Tie all the Gaussian amplitudes to the one of the target star
            fraction = fluxes[i]/fluxes[0]
            modified_tie_amplitude = partial(tie_amplitude, factor=fraction)
            getattr(model,f'amplitude_{i}').tied = modified_tie_amplitude
            # Untie and unfix for the target star
            if i==0:
                getattr(model,f'amplitude_{i}').min = 0.0
                getattr(model,f'x_stddev_{i}').tied = False
                getattr(model,f'amplitude_{i}').tied = False
    # Add a 2D plane to the model
    median_background_flux = np.median(image[mask_background])
    model += Planar2D(slope_x=0, slope_y=0, intercept=median_background_flux)
    # Make the fit
    (xsize,ysize) = image.shape
    y, x = np.mgrid[:xsize, :ysize]
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Fit = fitter(model,x,y, image)
        except RecursionError as e:
            err_msg = utils.print_err(str(e), prepend=prepend_err_msg)
            return None, err_msg
    # Results
    fitted_image = Fit(x,y)
    Plane = Fit[-1]
    TargetStar = Fit[0]
    if nGaussians > 1:
        Neighbours = []
        for i in range(1,nGaussians):
            Neighbours.append(Fit[i])
        Neighbours = np.sum(Neighbours)
    else:
        Neighbours = None
    # Find the significance of the non-homogeneous background
    v1 = (0,0,1)
    v2 = (Plane.slope_x.value, Plane.slope_y.value,-1)
    v1 /=  np.linalg.norm(v1)
    v2 /=  np.linalg.norm(v2)
    cos = np.dot(v1,v2)
    tan = np.sqrt(1-cos**2)/cos
    bkg_change = xsize*tan
    try:
        fraction_bkg_change = np.abs(bkg_change/Plane(xsize/2,ysize/2)[0])
    except TypeError:
        fraction_bkg_change = np.abs(bkg_change/Plane(xsize/2,ysize/2))
    # Find the flux contribution of neighbor stars/target star/background to the aperture mask
    neighbour_flux = np.sum( (Fit(x,y)-TargetStar(x,y)-Plane(x,y))[mask_aperture] )
    target_flux = np.sum( TargetStar(x,y)[mask_aperture] )
    bkg_flux = np.sum( Plane(x,y)[mask_aperture] )
    fraction_ap_contamination = neighbour_flux/target_flux 
    # In order to pickle, remove functions references in tie attribute of the fit components 
    TargetStar.y_stddev.tied = None
    if nGaussians > 2:
        try:
            for Gaussian in Neighbours:
                Gaussian.amplitude.tied = None
                Gaussian.x_stddev.tied = None
                Gaussian.y_stddev.tied = None
        except Exception as e:
            e_name = type(e).__name__
            print(f'Unknwon error within function `contamination`. Exception -> {e_name}: {e}.')
    elif nGaussians == 2:
        Neighbours.amplitude.tied = None
        Neighbours.x_stddev.tied = None
        Neighbours.y_stddev.tied = None
    # Store to info
    info.fit = SimpleNamespace()
    info.fit.fitted_image = fitted_image
    info.fit.Plane = Plane # Function
    info.fit.TargetStar = TargetStar # Function
    info.fit.Neighbours = Neighbours # Function
    info.fit.xPixel = x # Pixel coordinates
    info.fit.yPixel = y # Pixel coordinates
    info.fit.neighbour_flux_ap = neighbour_flux
    info.fit.target_flux_ap = target_flux
    info.fit.bkg_flux_ap = bkg_flux
    info.fit.fraction_contamination_ap = fraction_ap_contamination
    info.fit.fraction_bkg_change = fraction_bkg_change
    return fitted_image, err_msg

def refine_aperture(info,
                    wcs,
                    prepend_err_msg='',
                    thresholds=iter([7.5, 10, 15, 20, 30, 40, 50]),
                    arcsec_per_pixel=21*u.arcsec, # TESS CCD,
                    delta_mag=4):
    """
    Purpose:
        Find an aperture mask that only contains one source and only
        decresing pixels in flux from that source

    Args:
        info:
            SimpleNamespace object with the following attributes:
                * info.median_image
                * info.masks.aperture
                * info.masks.background
                * info.target.pix
                * info.target.mag
                * info.neighbours_used.pix
                * info.neighbours_used.mag
            to which will be added:
                * info.target
                * info.neighbours_all
                * info.neighbours_used
        wcs (astropy.wcs.WCS):
            World Coordinate System.
        prepend_err_msg (str, optional):
            String to prepend to the error message. Defaults to ''.
        thresholds (iter, optional):
            Iterator with the increasing aperture thresholds to be used in case
            there are neighbour stars within the aperture mask. The increasing
            thresolds are meant to decrease the size of the aperture mask. If
            no threshold leads to only the target star within the aperture mask,
            then the light curve extraction is halted. Defaults to
            iter([7.5, 10, 15, 20, 30, 40, 50]).
        arcsec_per_pixel (astropy.units.quantity.Quantity, optional):
            Size of a pixel in arcseconds. Defaults to 21*u.arcsec.
        delta_mag (int, optional):
            Magnitude difference between the target and the neighbours, ie,
            number of magnitudes dimmer than the target up to which neighbour
            stars will be considered. Defaults to 4.

    Returns:
        aperture (numpy.ndarray):
            New aperture mask.
        err_msg (str):
            Error message. If no error, it is an empty string.
    """
    # Unpack information
    tic = info.tic
    ra = info.ra
    dec = info.dec
    aperture = info.masks.aperture
    image = info.median_image
    threshold = info.aperture_threshold
    # Initialization
    err_msg = ''
    # Query surrounding area in MAST
    search_radius_pixel = np.sqrt(2*np.max(image.shape)**2)/2
    search_radius = search_radius_pixel * arcsec_per_pixel
    target_coord = SkyCoord(ra, dec, unit = "deg")
    tic_tmp,\
    tess_coord, target_tmag,\
    nb_coords, nb_tmags = query_TIC(f'TIC {tic}',target_coord, search_radius=search_radius, tic_id=tic)
    # Check we retrieve the correct target
    if tic != tic_tmp:
        err_msg = utils.print_err('The TIC number from the MAST query does not match the one from the TESS FITS image.', prepend=prepend_err_msg)
        print(err_msg)
        return None, err_msg
    # Make neighbor coordenates into NumPy arrays
    nb_coords = np.array(nb_coords)
    # Store to info (get plain numbers from the AstroPy instance)
    info.neighbours_all = SimpleNamespace()
    info.neighbours_all.mag = nb_tmags
    info.neighbours_all.ra = np.array([coord.ra.deg  for coord in nb_coords])
    info.neighbours_all.dec = np.array([coord.dec.deg for coord in nb_coords])
    # Filter neighbour stars: Remove too faint stars
    nb_faintest_mag = target_tmag + delta_mag
    mask = nb_tmags <= nb_faintest_mag
    nb_tmags =  nb_tmags[mask]
    nb_coords = nb_coords[mask]
    # Convert coordenates to pixels
    target_coord_pixel = np.array( [target_coord.to_pixel(wcs,origin=0)] )
    # Store to info (get plain numbers from the AstroPy instance)
    info.target = SimpleNamespace()
    info.target.mag = target_tmag
    info.target.ra = target_coord.ra.deg
    info.target.dec = target_coord.dec.deg
    info.target.pix = target_coord_pixel
    if nb_coords.size > 0:
        nb_coords_pixel = np.array( [coord.to_pixel(wcs,origin=0) for coord in nb_coords], dtype=float)
        # Filter neighbour stars: Remove stars outside the image bounds
        nb_pixel_value = ndimage.map_coordinates(image, [nb_coords_pixel[:,1], nb_coords_pixel[:,0]], order=0)
        mask = nb_pixel_value != 0
        nb_tmags =  nb_tmags[mask]
        nb_coords = nb_coords[mask]
        nb_coords_pixel = nb_coords_pixel[mask,:]
    else:
        nb_coords_pixel = np.array([])
    # Store to info (get plain numbers from the AstroPy instance)
    info.neighbours_used = SimpleNamespace()
    info.neighbours_used.mag = nb_tmags
    info.neighbours_used.ra = np.array([coord.ra.deg  for coord in nb_coords])
    info.neighbours_used.dec = np.array([coord.dec.deg for coord in nb_coords])
    info.neighbours_used.pix = nb_coords_pixel
    if nb_coords.size > 0:
        # Make neighbour pixels coordenates match the image grid, ie, bin them
        nb_coords_pixel_binned = np.floor(nb_coords_pixel+0.5)
        while True:
            # Find if a neighbour is within the aperture mask
            overlaps = ndimage.map_coordinates(aperture.astype(int), [nb_coords_pixel_binned[:,1], nb_coords_pixel_binned[:,0]], order=0)
            if overlaps.sum() == 0:
                break
            else:
                try:    
                    # Make a new aperture mask
                    threshold = next(thresholds)
                    aperture = threshold_mask(image, threshold=threshold, reference_pixel='center')
                except StopIteration:
                    # If no more thresholds to try, set the aperture to `None`
                    err_msg = utils.print_err('Not isolated target star.', prepend=prepend_err_msg)
                    info.masks.aperture = None
                    return None, err_msg
                if np.sum(aperture.astype(int)) == 0:
                    # If no aperture left, set the aperture to `None`
                    err_msg = utils.print_err('Not isolated target star.', prepend=prepend_err_msg)
                    info.masks.aperture = None
                    return None, err_msg
    # Store to info
    info.aperture_threshold = threshold
    # Find the brightest pixel within the mask
    ap_image = np.ma.masked_where(~aperture, image)
    seeds = np.argwhere(ap_image==ap_image.max())
    # Ensure the mask contains only pixels with decreasing flux w.r.t. the brightest pixel
    aperture = find_fainter_adjacent_pixels(seeds,ap_image)
    # Make target pixel coordenate match the image grid, ie, bin it
    target_coords_pixel_binned = np.floor(target_coord_pixel+0.5)
    # Find if a target is within the aperture mask
    overlaps = ndimage.map_coordinates(aperture.astype(int), [target_coords_pixel_binned[:,1], target_coords_pixel_binned[:,0]], order=0)
    if overlaps.sum() == 0:
        err_msg = utils.print_err('Target star not within the mask.', prepend=prepend_err_msg)
        print(err_msg)
        info.masks.aperture = None
        return None, err_msg
    # Store to info
    info.masks.aperture = aperture
    return aperture, err_msg

def exclude_intervals(tpf,
                      info,
                      intervals):
    """
    Purpose:
        Ser the attribute `quality_mask` of the TPF to FALSE for the given
        intervals.

    Args:
        tpf (lightkurve.targetpixelfile.TessTargetPixelFile):
            LightKurve target pixel file object.
        info:
            SimpleNamespace object with the following attributes:
                * info.sector
            to which will be added:
                * info.excluded_intervals
        intervals (dict):
            Dictionary indicating the intervals to exclude. The keys are
            integers indicating TESS sectors, and values are lists of tuples.
            Each tuple contains the initial and final time of the interval to
            exclude. Times must be given with astropy units.
            Example: dictionary to set to FALSE the quality mask in TESS sectors
            1 and 6 for the given intervals.
                > import astropy.units as u
                > intervals = {}
                > intervals[1] = [ (1334.8, 1335.1)*u.day,
                                   (1347.0, 1349.5)*u.day ]
                > intervals[6] = [ (1476.0, 1479.0)*u.day ]

    Returns:
        tpf (lightkurve.targetpixelfile.TessTargetPixelFile):
            Same lightKurve target pixel file object as in the argumant, but
            with the updated attribute `quality_mask`.
    """    
    # Validate the input
    if not isinstance(intervals, dict):
        raise TypeError('`intervals` must be a dictionary.')
    valid_types = (u.Unit, u.core.IrreducibleUnit, u.core.CompositeUnit, u.quantity.Quantity)
    for key, value in intervals.items():
        if not isinstance(key,int):
            raise TypeError('`intervals` keys must be integers indicating the corresponding TESS sector.')
        conditions = [isinstance(value,list),
                      all(len(interval)==2 for interval in value),
                      all(isinstance(interval[0],valid_types) for interval in value),
                      all(isinstance(interval[1],valid_types) for interval in value)]
        if not all(conditions):
            raise TypeError('`intervals` values must be lists of tuples indicating the intervals to exclude. Each tuple must have two elements: the start and end of the interval expressed as a `~astropy.units.Quantity` or a `~astropy.units.Unit`.')
    
    if info.sector in intervals.keys():
        intervals = intervals[info.sector]
        # Format accordingly to the TPF time
        intervals = [Time(interval, format=tpf.time.format, scale=tpf.time.scale) for interval in intervals]
        for interval in intervals:
            # Find the indices of the quality mask that created tpf.time
            ind = np.argwhere(tpf.quality_mask == True)
            mask  = tpf.time > interval[0]
            mask &= tpf.time < interval[1]
            # Set to False the indices to be masked (ignored).
            # Note that we take a subset from `ind` because the masks where defined from tpf.time
            tpf.quality_mask[ind[mask]] = False
        # Store to info
        # info.excludedintervals = np.array(_intervals)
        info.excluded_intervals = intervals
    else:
        # Store to info
        info.excluded_intervals = None
    return tpf

def find_number_of_PCs(info,
                       regressors,
                       lc,
                       npc=7,
                       nbins=40,
                       threshold_variance=1e-4):
    """
    Purpose:
        Find the number of principal components to use in the PCA method

    Args:
        info:
            SimpleNamespace to which will be added:
                * info.pca_all.pca_all
                * info.pca_all.coef
                * info.pca_all.pc
                * info.pca_all.dm
                * info.pca_all.rc
                * info.pca_all.npc
                * info.pca_all.npc_used
                * info.pca_all.pc_variances
                * info.pca_all.threshold_variance
                * info.pca_all.nbins
        regressors (numpy.ndarray): 
            Regressors to be used for the design matrix.
        lc (lightkurve.lightcurve.TessLightCurve):
            Light curve.
        npc (int, optional): 
            Maximum number of principal components to use. Defaults to 7.
        nbins (int, optional):
            Number of bins used to chop the principal components in smaller
            light curves of equal lenght. These smaller light curves will be
            used to find the level of scatter (variance) in the light curve.
            Defaults to 40.
        threshold_variance (float, optional):
            Threshold variance to use to find the number of principal
            components. Principal components with variance above this threshold
            are not considered. Defaults to 1e-4.
    
    Returns:
    
        new_npc (int):
            Number of principal components whose variance is below the
            threshold.
        dm (lightkurve design matrix object):
            lightkurve design matrix object.
        rc (lightkurve regression corrector object):
            lightkurve regression corrector object.
    """
    dm = lk.DesignMatrix(regressors, name='regressors').pca(npc).append_constant()
    rc = lk.RegressionCorrector(lc)
    lc_regressed = rc.correct(dm)
    # Find the median of the moving variance for each PCs
    boxsize = np.floor( dm.values[:,0].size/nbins ).astype(int)
    pcs_variances = []
    for ipc in range(npc):
        pcs_variances.append( pd.Series(dm.values[:,ipc]).rolling(boxsize).var().median() )
    # pcs_variances = [ pd.Series(dm.values[:,ipc]).rolling(boxsize).var().median() for ipc in range(npc) ]
    pcs_variances = np.array(pcs_variances)
    relative_change_variances = np.abs(np.diff(pcs_variances)/pcs_variances[:-1]) # < THIS IS NOT USED.
    # Select PCs with variance above threshold value
    ind_bad_pcs = np.argwhere(pcs_variances>threshold_variance)
    if ind_bad_pcs.size > 0:
        # Find index first PC that exceeds threshold value. This is the new npc
        new_npc = ind_bad_pcs[0].item()
    else:
        print(f'TIC {info.tic} Sector {info.sector}: No principal components (PCs) with variance>{threshold_variance}. All {npc} PCs used (this is just FYI, for further information see arguments `max_num_of_pc`, `num_of_pc_bins`, `pc_threshold_variance`.')
        new_npc = npc
    # Store to info
    info.pca_all = SimpleNamespace()
    info.pca_all.coef = rc.coefficients
    info.pca_all.pc = [dm.values[:,i] for i in range(dm.rank)]
    info.pca_all.dm = dm
    info.pca_all.rc = rc
    info.pca_all.npc = npc
    info.pca_all.npc_used = new_npc
    info.pca_all.pc_variances = pcs_variances
    info.pca_all.threshold_variance = threshold_variance
    info.pca_all.nbins = nbins
    return new_npc, dm, rc

def create_output_structure():
    """Dictionary-like object that organize the output of the pipeline."""
    info = SimpleNamespace()
    info.tic = None
    info.sector = None
    info.ra = None
    info.dec = None
    info.headers = None
    info.fit = None
    info.neighbours_all = None
    info.neighbours_used = None
    info.target = None
    info.aperture_threshold = None
    info.pca_all = None
    info.pca_used = None
    info.centroids = None
    info.excluded_intervals = None
    info.lc_raw1 = None
    info.lc_raw2 = None
    info.lc_trend = None
    info.lc_regressed = None
    info.lc_regressed_clean = None
    info.median_image = None
    info.masks = None
    info.tag = None
    return info

def extract_light_curve(fitsFile,
                        outputdir='processed',
                        return_msg=True,
                        overwrite=False,
                        progressbar=False,
                        ncores=1,
                        verbose=False,
                        excluded_intervals=None,
                        delta_mag=4,
                        arcsec_per_pixel=21*u.arcsec,
                        aperture_mask_threshold=5,
                        background_mask_threshold=3,
                        aperture_mask_increasing_thresholds=iter([7.5, 10, 15, 20, 30, 40, 50]),
                        max_num_of_pc=7,
                        num_of_pc_bins=40,
                        pc_threshold_variance=1e-4,
                        sigma_clipping=5,
                        aperture_mask_min_pixels=4,
                        aperture_mask_max_elongation=14):
    """
    Purpose:
        Extract light curve from a TESS Target Pixel File (TPF).

    Args:
        fitsFile (str | list[str]):
            Path to the TESS TPF to process. It can also be a list of paths,
            including numpy.ndarray and pandas.core.series.Series.
        outputdir (str, optional):
            Output directory to save the processed light curves as pickle
            files. Defaults to 'processed'.
        return_msg (bool, optional):
            Whether to return a success message about the reduction process.
            Defaults to True.
        overwrite (bool, optional):
            Whether to overwrite the output file if it already exists. Defaults
            to False.
        progressbar (bool, optional):
            Whether to show a progress bar. Defaults to False.
        ncores (int, optional):
            Number of cores to use. Defaults to 1. If set to None, use all
            available cores.
        verbose (bool, optional):
            Whether to print additional messages. Defaults to False.
        excluded_intervals (dict):
            Dictionary indicating the intervals to exclude. The keys are
            integers indicating the TESS sectors, and the values are lists of
            tuples, each tuple containing an initial and final time to exclude.
            Times must be given with astropy units.
            Example: dictionary to set to FALSE the quality mask in TESS sectors
            1 and 6 for the given intervals.
                > import astropy.units as u
                > intervals = {}
                > intervals[1] = [ (1334.8, 1335.1)*u.day,
                                   (1347.0, 1349.5)*u.day ]
                > intervals[6] = [ (1476.0, 1479.0)*u.day ]
        delta_mag (int, optional):
            Magnitude difference between the target and the neighbours, ie,
            number of magnitudes dimmer than the target up to which neighbour
            stars will be considered. Defaults to 4.
        arcsec_per_pixel (astropy.units.quantity.Quantity, optional):
            Size of a pixel in arcseconds. Defaults to 21*u.arcsec.
        aperture_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        background_mask_threshold : float
            A value for the number of sigma by which a pixel needs to be
            dimmer than the median flux to be included in the background mask.
        aperture_mask_increasing_thresholds (iter, optional):
            Iterator with the increasing aperture thresholds to be used in case
            there are neighbour stars within the aperture mask. The increasing
            thresolds are meant to decrease the size of the aperture mask. If
            no threshold leads to only the target star within the aperture mask,
            then the light curve extraction is halted. Defaults to
            iter([7.5, 10, 15, 20, 30, 40, 50]).
        max_num_of_pc (int, optional): 
            Maximum number of principal components to use. Defaults to 7.
        num_of_pc_bins (int, optional):
            Number of bins used to chop the principal components in smaller
            light curves of equal lenght. These smaller light curves will be
            used to find the level of scatter (variance) in the light curve.
            Defaults to 40.
        pc_threshold_variance (float, optional):
            Threshold variance to use to find the number of principal
            components. Principal components with variance above this threshold
            are not considered. Defaults to 1e-4.
        sigma_clipping (float, optional):
            Sigma clipping value to be applied to the flux of the light curve
            after the detrending. Defaults to 5.
        aperture_mask_min_pixels (int, optional):
            Minimum number of pixels in the aperture mask. Note that the
            aperture mask can end up smaller than `aperture_mask_min_pixels` while attempting
            to isolate the target star from neighbour stars. Default is 4.
        aperture_mask_max_elongation (int, optional):
            Triggers when a 2D aperture mask has 4 or less columns (rows) and at
            least one row (column) larger that `aperture_mask_max_elongation` pixels. In such
            a case, the aperture mask in considered corrupted and the light
            curve extraction is halted. Default is 14.

    Returns:
        None
        
    Write:
        A pickle file containing a SimpleNamespace object with the attributes
        listed below. If there is a problem during the light curve extraction (or
        not need for some of the attributes), they will be set to None or numpy.nan.
        
        1. tic:     (int)
        TIC number.
        2. sector:  (int)
        TESS sector number.
        3. ra:      (float)
        Right ascension.
        4. dec:     (float)
        Declination.
        5. headers: (list)
        List with all headers in the original TPF.
        
        6. fit: (types.SimpleNamespace)
        Information regarding the fit of 2D Gaussians reproducing the TPF's median cadence image.
        |--- 6.1. fitted_image:               (numpy.ndarray)
        |         TPF's median cadence image.
        |--- 6.2. Plane:                      (astropy.modeling.functional_models.Planar2D)
        |         Fitted model (2D plane) of the background signal of the `fitted_image`. 
        |--- 6.3. TargetStar:                 (astropy.modeling.functional_models.Gaussian2D)
        |         Fitted model (2D Gaussian) of the target star in the `fitted_image`. 
        |--- 6.4. Neighbours:                 (astropy.modeling.core.CompoundModel)
        |         Fitted model (set of 2D Gaussians) of the neighbouring stars in the `fitted_image`. 
        |--- 6.5. xPixel:                     (numpy.ndarra>)
        |         Domain of `fitted_image` expresed as pixel coordinates. Generated by `numpy.mgrid` function. X-like coordinates.
        |--- 6.6. yPixel:                     (numpy.ndarra>)
        |         Domain of `fitted_image` expresed as pixel coordinates. Generated by `numpy.mgrid` function. Y-like coordinates.
        |--- 6.7. neighbour_flux_ap:          (numpy.float64)
        |         Flux contribution of neighboring stars to the aperture mask.
        |--- 6.8. target_flux_ap:             (numpy.float64)
        |         Flux contribution of target star to the aperture mask.
        |--- 6.9. bkg_flux_ap:                (numpy.float64)
        |         Flux contribution of background to the aperture mask.
        |--- 6.10. fraction_contamination_ap: (numpy.float64)
        |          Flux ratio of the aperture mask from neighbouring stars and the target star. I.e., `neighbour_flux/target_flux `.
        |--- 6.11. fraction_bkg_change:       (numpy.float64)
        |          Flux ratio of the background maximun change and average background.

        7. neighbours_all: (types.SimpleNamespace)
        Information regarding all found neighbouring stars.
        |--- 7.1. mag: (numpy.ndarray)
        |         TESS magnitude.
        |--- 7.2. ra:  (numpy.ndarray)
        |         Right ascension.
        |--- 7.3. dec: (numpy.ndarray)
        |         Declination.

        8. neighbours_used: (types.SimpleNamespace)
        Information regarding only neighbouring stars used during the fit.
        |--- 8.1. mag: (numpy.ndarray)
        |         TESS magnitude.
        |--- 8.2. ra:  (numpy.ndarray)
        |         Right ascension.
        |--- 8.3. dec: (numpy.ndarray)
        |         Declination.
        |--- 8.4. pix: (numpy.ndarray)
        |         Pixel coordinates.

        9. target: (types.SimpleNamespace)
        |--- 9.1. mag: (numpy.float64)
        |         Description
        |--- 9.2. ra:  (numpy.float64)
        |         Description
        |--- 9.3. dec: (numpy.float64)
        |         Description
        |--- 9.4. pix: (numpy.ndarray)
        |         Description

        10. aperture_threshold: (int)
            Number of sigma the aperture mask is brighter than the median flux of the TPF's median cadence image.
            
        11. pca_all: (types.SimpleNamespace)
            Information regarding all principal components available to detrend the light curve.
        |--- 11.1. coef:               (numpy.ndarray)
        |          Coefficients of the principal components.
        |--- 11.2. pc:                 (list)
        |          Principal components.
        |--- 11.3. dm:                 (lightkurve.correctors.designmatrix.DesignMatrix)
        |          Design matirx for use in linear regression. A matrix of column vectors (principal components) used for the linear regression.
        |--- 11.4. rc:                 (lightkurve.correctors.regressioncorrector.RegressionCorrector)
        |          Regression Corrector object used to remove noise using linear regression against a design matrix `dm`.
        |--- 11.5. npc:                (int)
        |          Number of principal components available.
        |--- 11.6. npc_used:           (int)
        |          Number of principal components used for the linear regression.
        |--- 11.7. pc_variances:       (numpy.ndarray)
        |          Estimate level of variance of each normalized principal component. Calculated as the median variance of chopped principal componentes (partitions).
        |--- 11.8. threshold_variance: (float)
        |          Principal components with `pc_variances` values above this are not considered for the linear regression. 
        |--- 11.9. nbins:              (int)
        |          Number of partition each principal component is divided to calculate 'pc_variances`.

        12. pca_used: (types.SimpleNamespace)
            Information regarding only principal components used for detrending of the light curve.
        |--- 12.1. coef: (numpy.ndarray)
        |          Coefficients of the principal components.
        |--- 12.2. pc:   (list)
        |          Principal components.
        |--- 12.3. dm:   (lightkurve.correctors.designmatrix.DesignMatrix)
        |          Design matirx for use in linear regression. A matrix of column vectors (principal components) used for the linear regression.
        |--- 12.4. rc:   (lightkurve.correctors.regressioncorrector.RegressionCorrector)
        |          Regression Corrector object used to remove noise using linear regression against a design matrix `dm`.
        |--- 12.5. npc:  (int)
        |          Number of principal components.

        13. centroids:             (types.SimpleNamespace)
            Information regarding the centroid of the TPF's median cadence image.
        |--- 13.1. col:            (numpy.ndarray)
        |          Its column-like location.
        |--- 13.2. row:            (numpy.ndarray)
        |          Its row-like location.
        |--- 13.3. sqrt_col2_row2: (numpy.ndarray)
        |          Its distance-like location.
        |--- 13.4. time:           (numpy.ndarray)
        |          Time of the corresponding cadence.

        14. excluded_intervals: (dict)
            Intervals to exclude. The keys are integers indicating TESS sectors, and values are lists of tuples. Each tuple contains the initial and final time of the interval to exclude. Times are astropy units.
        15. lc_raw1:            (lightkurve.lightcurve.TessLightCurve)
            Raw light curve, without excluded intervals, generated using simple aperture potometry with the aperture mask.
        16. lc_raw2:            (lightkurve.lightcurve.TessLightCurve)
            Raw light curve, with excluded intervals, generated using simple aperture potometry with the aperture mask.
        17. lc_trend:           (lightkurve.lightcurve.LightCurve)
            Regresor light curve obtained from the principal component analysis. Used for the linear regression of `lc_raw2`.

        18. lc_regressed: (types.SimpleNamespace)
            Light curve after the linear regression. I.e., light curve with systematics corrected.
        |--- 18.1. lc:             (lightkurve.lightcurve.TessLightCurve)
        |          Light curve.
        |--- 18.2. outlier_mask:   (numpy.ndarray)
        |          Mask indicating values marked for removal by the sigma clipping. 
        |--- 18.3. sigma_clipping: (int)
        |          Sigma value used by the sigma clipping method on the light curve flux.

        19. lc_regressed_clean: (lightkurve.lightcurve.TessLightCurve)
            Regressed light curve with flux outliers removed by sigma clipping.
        20. median_image:            (numpy.ndarray)
            TPF's median cadence image.
            
        21. masks: (types.SimpleNamespace)
            Information regarding the masks for the TPF's median cadence image.
        |--- 21.1. aperture:   (numpy.ndarray)
        |          Aperture mask.
        |--- 21.2. background: (numpy.ndarray)
        |          Background mask.

        22. tag: (str)
            Sentence aimed to identify potential problems during the light curve extraction.
    """
    # Handle a list-like input
    valid_types = (list,np.ndarray,pd.core.series.Series)
    if isinstance(fitsFile,valid_types):
        # Update kw arguments of the function
        _extract_light_curve = partial(extract_light_curve, outputdir=outputdir,
                                                            return_msg=return_msg,
                                                            overwrite=overwrite,
                                                            progressbar=False,
                                                            ncores=1,
                                                            verbose=verbose,
                                                            excluded_intervals=excluded_intervals,
                                                            delta_mag=delta_mag,
                                                            arcsec_per_pixel=arcsec_per_pixel,
                                                            aperture_mask_threshold=aperture_mask_threshold,
                                                            aperture_mask_increasing_thresholds=aperture_mask_increasing_thresholds,
                                                            max_num_of_pc=max_num_of_pc,
                                                            num_of_pc_bins=num_of_pc_bins,
                                                            pc_threshold_variance=pc_threshold_variance,
                                                            sigma_clipping=sigma_clipping,
                                                            aperture_mask_min_pixels=aperture_mask_min_pixels,
                                                            aperture_mask_max_elongation=aperture_mask_max_elongation)
        # Use a simple for loop (to avoid multiprocessing issues)
        if ncores==1:
            for fitsfile in fitsFile:
                _extract_light_curve(fitsfile)
        else:
            # Pool for parallel processing
            with Pool(ncores) as pool:
                it = pool.imap_unordered(_extract_light_curve, fitsFile)
                if progressbar:
                    it = tqdm(it, total=len(fitsFile))
                # Exhaust the iterator
                deque(it, maxlen=0)
        return
    
    if not isinstance(fitsFile,str):
        raise TypeError('`fitsFile` must be a string pointing to the TESS Target Pixel File to process. It can also be a list of those, including numpy.ndarray and pandas.core.series.Series.')
            
    fitsFile = Path(fitsFile)
    # Print name of file being processed
    if verbose:
        print(f'Working on {fitsFile.name}')  
    # Check i/o directories and files
    if not (outputdir := Path(outputdir)).exists():
        outputdir.mkdir(parents=True)
    outputname = Path(fitsFile).stem+'_corrected.pickle'
    output = outputdir/outputname
    # Check if output file already exists
    if output.exists() and not overwrite:
        print(f'Skipped: Output file already exists: {output.name}.')
        return
    # Structure the data to be saved
    results = create_output_structure()
    # Save headers from original FITS file
    results.headers = utils.get_header_info(fitsFile)
    # Load the TESS target pixel file
    try:
        tpf = lk.TessTargetPixelFile(fitsFile)
    except Exception as e:
        # Save results
        e_name = e.__class__.__name__
        err_msg = f'"lightkurve.TessTargetPixelFile()" could not open file {fitsFile}. Exception: -> {e_name}: {e}.'
        print(err_msg)
        results.tag = err_msg
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    # Store to results
    results.tic = tpf.get_keyword('ticid')
    results.sector = tpf.get_keyword('sector')
    results.ra = tpf.ra 
    results.dec = tpf.dec
    # Initialize messages
    id_msg = f'TIC {results.tic} Sector {results.sector}: Skipped: '
    OK_msg = f'TIC {results.tic} Sector {results.sector}: OK'
    # Calculate the median image (based on function `create_threshold_mask` from `lightkurve` package).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results.median_image = np.nanmedian(tpf.flux, axis=0).value
    # Estimate of aperture mask and background mask
    results.aperture_threshold = aperture_mask_threshold
    aperture_mask = threshold_mask(results.median_image, threshold=aperture_mask_threshold, reference_pixel='center')
    background_mask = ~ threshold_mask(results.median_image, threshold=background_mask_threshold, reference_pixel=None)
    # Exclude NaN values outside the camera
    background_mask &= ~np.isnan(results.median_image) 
    # Estimate the median flux background
    median_background_flux = np.median(results.median_image[background_mask])
    # Store to results
    results.masks = SimpleNamespace()
    results.masks.aperture = aperture_mask
    results.masks.background = background_mask
    # Check validity of aperture mask
    OK_ap_mask, err_msg = check_aperture_mask(results.masks.aperture,
                                              prepend_err_msg=id_msg,
                                              aperture_mask_min_pixels=aperture_mask_min_pixels,
                                              aperture_mask_max_elongation=aperture_mask_max_elongation)
    # If aperture is not good, exit program with corresponding message
    if not OK_ap_mask:
        # Save results
        results.tag = err_msg
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    # Refine aperture
    try:
        WCS = tpf.wcs
    except IndexError:
        # Save results
        err_msg = id_msg+'No WCS info in header'
        print(err_msg)
        results.tag = err_msg
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    results.masks.aperture, err_msg = refine_aperture(results,
                                                      WCS,
                                                      prepend_err_msg=id_msg,
                                                      delta_mag=delta_mag,
                                                      arcsec_per_pixel=arcsec_per_pixel,
                                                      thresholds=aperture_mask_increasing_thresholds)
    # If not satisfactory aperture mask
    if results.masks.aperture is None:
        # Save results
        results.tag = err_msg
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    # Variation in time of aperture's center of mass
    centroid_col, centroid_row = tpf.estimate_centroids(aperture_mask=results.masks.aperture, method='quadratic')
    centroid_col = centroid_col.value - tpf.column
    centroid_row = centroid_row.value - tpf.row
    sqrt_col2_row2 = np.sqrt(centroid_col**2+centroid_row**2)
    # Store to results
    results.centroids = SimpleNamespace(col=centroid_col,
                                        row=centroid_row,
                                        sqrt_col2_row2=sqrt_col2_row2,
                                        time=tpf.time.value)
    # Fit the image and find the contamination fraction within the aperture mask
    fitted_image, err_msg = contamination(results, prepend_err_msg=id_msg)
    if fitted_image is None:
        # Save results
        results.tag = err_msg
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    # Generate the raw light curve
    lc_raw1 = tpf.to_lightcurve(aperture_mask=results.masks.aperture, method='aperture')
    # Store to results
    results.lc_raw1 = lc_raw1
    # Find the indices of the quality mask that created the light curve
    ind = np.argwhere(tpf.quality_mask == True)
    # Masks with True value the light curve times with null or NaN flux
    mask  = lc_raw1.flux.value == 0
    mask |= lc_raw1.flux.value == np.nan
    # Set to False the indices to be masked (ignored).
    # Note that we take a subset from `ind` because the masks where defined from the light curve
    tpf.quality_mask[ind[mask]] = False
    if excluded_intervals is not None:
        tpf = exclude_intervals(tpf, results, excluded_intervals)
    # Generate the Simple-Aperture-Photometry light curve
    lc_sap = tpf.to_lightcurve(aperture_mask=results.masks.aperture, method='aperture')
    # Identify NaN values in the flux
    nan_mask = np.isnan(lc_sap.flux)
    # Remove NaN values
    lc_sap = lc_sap[~nan_mask]
    # Store to results
    results.lc_raw2 = lc_sap
    # Make a design matrix and pass it to a linear regression corrector
    regressors = tpf[~nan_mask].flux[:, background_mask]
    # Number of PCs to use
    npc, dm, rc = find_number_of_PCs(results,
                                     regressors,
                                     lc_sap,
                                     npc=max_num_of_pc,
                                     nbins=num_of_pc_bins,
                                     threshold_variance=pc_threshold_variance)
    if npc == 0:
        # Save results
        results.tag = id_msg+'None PC used, no detrended done.'
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return err_msg
        return
    try:
        # Detrend light curve using PCA
        dm = lk.DesignMatrix(regressors, name='regressors').pca(npc).append_constant()
        rc = lk.RegressionCorrector(lc_sap)
        lc_regressed = rc.correct(dm)
        lc_trend = rc.diagnostic_lightcurves['regressors']
        # Sigma clipping the remove outliers
        lc_regressed_no_outliers, lc_mask_regressed_outliers = lc_regressed.remove_outliers(return_mask=True, sigma=sigma_clipping)
        # Store to results
        results.lc_trend = lc_trend
        results.lc_regressed = SimpleNamespace()
        results.lc_regressed.lc = lc_regressed         
        results.lc_regressed.outlier_mask = lc_mask_regressed_outliers
        results.lc_regressed.sigma_clipping = sigma_clipping
        results.lc_regressed_clean = lc_regressed_no_outliers
        results.pca_used = SimpleNamespace()
        results.pca_used.coef = rc.coefficients
        results.pca_used.pc = [dm.values[:,i] for i in range(dm.rank)]
        results.pca_used.dm = dm
        results.pca_used.rc = rc
        results.pca_used.npc = npc
        # Save results
        results.tag = 'OK'
        with open(output,'wb') as picklefile:
            pickle.dump(results,picklefile)
        if return_msg:
            return OK_msg
        return
    except Exception as e:
        e_name = type(e).__name__
        print(f'!!! Sector {results.sector}. Unexpected EXCEPTION -> {e_name}: {e}.')
        if return_msg:
            return id_msg+'::'+repr(e)+'::'+str(e)
        return

def group_lcs(inputdir,
              outputdir='groupped',
              namePattern = 'tic{TIC}_sec{SECTOR}_corrected.pickle',
              outputname_pattern = 'tic{TIC}_allsectors_corrected.pickle',
              TICs='all',
              sectors='all',
              progressbar=False,
              ncores=-1):
    """
    Purpose:
        Group outputs from function `extract_light_curve` for different TESS
        sectors into a single pickle file. The grouping is based on the file name
        of the outputs of `extract_light_curve` function, ie, the TIC number and
        sector number are extracted from the file name.

    Args:
        inputdir (str):
            Input directory where the outputs of `extract_light_curve` are stored.
        outputdir (str, optional):
            Output directory where the groupped light curves will be stored.
            Defaults to 'groupped'.
        namePattern (str, optional):
            File name pattern used to search for the outputs of `extract_light_curve`.
            It must contain the strings '{TIC}' and '{SECTOR}' to indicate the 
            position of the TIC number and sector number in the file name.
            Defaults to 'tic{TIC}_sec{SECTOR}_corrected.pickle'.
        outputname_pattern (str, optional):
            File name pattern used to save the new pickle files for each TIC star.
            It must contain the strings '{TIC}' which will automatically be replaced
            for the TIC number. Defaults to 'tic{TIC}_allsectors_corrected.pickle'.
        TICs ("all" | int | list[int], optional):
            TIC numbers to group. The information about TIC number is extracted
            from the filename of the input pickle file. Defaults to 'all'.
        sectors (list[int], optional):
            Sectors to consider for the groupping. Defaults to 'all' groups all
            found sectors.
        progressbar (bool, optional):
            Show a progress bar. Defaults to False.
        ncores (int, optional):
            Number of parallel processes to group a list of TIC numbers. All
            sectors available for a TIC number are under one process, i.e., the
            parallelization happens at TIC level and not at sector level.
            Defaults to -1 max out available cores.
    
    Returns:
        None
    """

    # Check for input/output directories
    if not (inputdir := Path(inputdir)).exists():
        raise ValueError(f'Input directory {inputdir} does not exist.')
    if not (outputdir := Path(outputdir)).exists():
        outputdir.mkdir()
    # Get the filepaths and filenames of the files to process
    utils.contains_TIC_and_sector(namePattern)
    globPattern = namePattern.replace('{TIC}', '*').replace('{SECTOR}', '*')
    filepaths = [file for file in inputdir.glob(globPattern)]
    filenames = [file.name for file in filepaths]
    filesTable = pd.DataFrame({'filename':filenames, 'filepath':filepaths})
    # Get TIC and sector number from filename
    filesTable['tic'] = filesTable['filename'].apply(utils.return_TIC_1, args=(namePattern,))
    filesTable['sector'] = filesTable['filename'].apply(utils.return_sector, args=(namePattern,))
    # Sort by TIC and sector number
    filesTable.sort_values(by=['tic','sector'], inplace=True)
    # Select the TIC number to process
    if not TICs == 'all':
        if isinstance(TICs,int):
            filesTable = filesTable.query('tic == @TICs')  
        elif isinstance(TICs,(list,np.ndarray)):
            filesTable = filesTable.query('tic in @TICs')  
        else:
            raise ValueError(f'TICs must be "all", an integer or a list of integers. Got {TICs}')
    # Select the sectors to process
    if not sectors == 'all':
        if isinstance(sectors,int):
            filesTable = filesTable.query('sector == @sectors')  
        elif isinstance(sectors,(list,np.ndarray)):
            filesTable = filesTable.query('sector in @sectors')
        else:
            raise ValueError(f'`sectors` must be "all", an integer or a list of integers. Got {sectors}.')
    # Group filenames by the TIC number
    groups = filesTable.groupby('tic')
    def grouping(tic, group):
        """Unpickle all files for a TIC number, group them into a list, and pickle it."""
        # group = groups.get_group(tic)
        all_sectors = [] 
        # Loop over each sector in the group
        for sector in group.iloc:
            # Unpickle
            with open((file:=sector['filepath']),'rb') as picklefile:
                try:
                    result = pickle.load(picklefile)
                except EOFError as e:
                    print(f'Skipped: TIC={tic}, sector={sector.sector} -> file {file} seems to be empty.')
            all_sectors.append(result)
        # Save
        utils.contains_TIC(outputname_pattern)
        name = Path(outputname_pattern.format(TIC=tic))
        with open(outputdir/name, 'wb') as picklefile: 
            pickle.dump(all_sectors, picklefile)
    # Enable parallel processing
    if progressbar:
        groups = tqdm(groups)
    Parallel(n_jobs=ncores)(delayed(grouping)(tic,group) for tic,group in groups)

def stitch_group(inputdir,
                 TICs='all',
                 namePattern='tic{TIC}_allsectors_corrected.pickle',
                 outputdir='stitched',
                 outputname_pattern='lc_tic{TIC}_corrected_stitched.csv',
                 progressbar=False,
                 ncores=1,
                 overwrite=False):
    """
    Purpose:
        Stitch the light curves contained in the outputs from function `group_lcs`
        within the given input directory. Return a CSV file per TIC star. The search
        of input files within the given input directory is based on file names in
        such a directory.

    Args:
        inputdir (str):
            Input directory where the outputs of `group_lcs` are stored.
        TICs ("all" | int | list[int], optional):
            TIC numbers to group. The information about TIC number is extracted
            from the filename of the input pickle file. Defaults to 'all'.
        namePattern (str, optional):
            File name pattern used to search for the outputs of `group_lcs`.
            It must contain the strings '{TIC}' to indicate the position of the
            TIC number in the file name. Defaults to 'tic{TIC}_allsectors_corrected.pickle'.
        outputdir (str, optional):
            _description_. Defaults to 'stitched'.
        outputname_pattern (str, optional):
            File name pattern used to save the CSV files for each TIC star.
            It must contain the strings '{TIC}' which will automatically be replaced
            for the TIC number. Defaults to 'lc_tic{TIC}_corrected_stitched.csv'.
        progressbar (bool, optional):
            Show a progress bar. Defaults to False.
        ncores (int, optional):
            Number of parallel processes to stitch a list of TIC numbers. All
            sectors available for a TIC number are under one process, i.e., the
            parallelization happens at TIC level and not at sector level.
            Defaults to -1 max out available cores.
        overwrite (bool, optional):
            Whether to overwrite results. Defaults to False.

    Returns:
        None
    """
    # Check for input/output directories
    if not (inputdir := Path(inputdir)).exists():
        raise ValueError(f'Input directory {inputdir} does not exist.')
    if not (outputdir := Path(outputdir)).exists():
        outputdir.mkdir()
    # Get the filepaths and filenames of the files to process
    utils.contains_TIC(namePattern)
    globPattern = namePattern.replace('{TIC}', '*')
    filepaths = [file for file in inputdir.glob(globPattern)]
    filenames = [file.name for file in filepaths]
    filesTable = pd.DataFrame({'filename':filenames, 'filepath':filepaths})
    # Get TIC number from filename
    filesTable['tic'] = filesTable['filename'].apply(utils.return_TIC_2, args=(namePattern,))
    # Sort by TIC number
    filesTable.sort_values(by=['tic'], inplace=True)
    # Select the TIC number to process
    if not TICs == 'all':
        if isinstance(TICs,int):
            filesTable = filesTable.query('tic == @TICs')  
        elif isinstance(TICs,(list,np.ndarray)):
            filesTable = filesTable.query('tic in @TICs')  
        else:
            raise ValueError(f'TICs must be "all", an integer or a list of integers. Got {TICs}')
    # Group filenames by the TIC number
    groups = filesTable.groupby('tic')
    def stitching(TIC):
        # Pickle file containing LC of all sectors
        file = inputdir/Path(namePattern.format(TIC=TIC))
        # Read pickle file
        with open(file, 'rb') as picklefile:
            all_sectors = pickle.load(picklefile)
        # Read the results
        lcs = [sector.lc_regressed_clean for sector in all_sectors if sector.tag == 'OK']
        # If no OK sectors 
        if len(lcs) == 0:
            return None
        # Sticht the light curve
        lc = lk.LightCurveCollection(lcs).stitch(corrector_func=normalize_lightCurve)        
        # Save LC as CSV file
        NameOutput_StitchedLC = Path(outputname_pattern.format(TIC=TIC))
        outputname = outputdir/NameOutput_StitchedLC
        if outputname.exists() and not overwrite:
            print(f'Skipped: File {outputname.name} already exists.')
        else:
            lc.to_csv(outputname)
    # Enable parallel processing
    TICs = groups.groups.keys()
    if progressbar:
        TICs = tqdm(TICs)
    Parallel(n_jobs=ncores)(delayed(stitching)(tic) for tic in TICs)

def get_group_summary(files,
                      csvname='summary.csv',
                      namePattern='tic{TIC}_allsectors_corrected.pickle',
                      TICs='all',
                      progressbar=False,
                      overwrite=False):
    """
    Purpose:
        Generte a CSV table from the output of `group_lcs` with information
        characterizing the light curve extraction for each TPF processed. The
        CSV table contains the following columns:
            * tic: TIC number.
            * sector: Sector number.
            * ra: Right ascension.
            * dec: Declination.
            * flux_contamination_fraction: Fraction of flux contamination in the aperture mask.
            * background_change_fraction: Fraction of flux background change.
            * mag: TESS magnitude.
            * number_of_pc_used: Number of principal components used for the correction of the light curve.
            * aperture_mask_threshold: Threshold used to create the aperture mask.
            * aperture_mask_size: Number of pixels in the aperture mask.
            * background_mask_size: Number of pixels in the background mask.
            * tag: Message about the light curve extraction.
            * time_span: Time span of the light curve in days.
            * time_points: Number of time points (cadences) in the light curve.
         
    Args:
        files (str | list[str]):
            Path to pickle file created by `group_lcs`. It can also be a list of
            those, including numpy.ndarray and pandas.core.series.Series.
            Alternatively, the path to a directory containing the pickle files
            is also accepted, in which case all pickle files in that directory are
            used (those files are found besed on the `namePattern` argument).
        TICs ("all" | int | list[int], optional):
            TIC numbers to consider. The information about TIC number is extracted
            from the filename of the input pickle file. Defaults to 'all'.
        namePattern (str, optional):
            File name pattern used to search for the outputs of `group_lcs` in case
            that `files` is a directory. It must contain the strings '{TIC}' to
            indicate the position of the TIC number in the file name.
            Defaults to 'tic{TIC}_allsectors_corrected.pickle'.
        csvname (str, optional):
            File name used to save the CSV files. Defaults to 'summary.csv'.
        progressbar (bool, optional):
            Show a progress bar. Defaults to False.
        overwrite (bool, optional):
            Whether to overwrite results. Defaults to False.

    Returns:
        None
    """

    def read_pickle_file(file, verbose=True):
        """Read pickle file created by function `group_lcs`."""
        error_message = '`file` must be a str or pathlib.Path pointing to the pickle file created by `group_lcs`.'
        if isinstance(file,str):
            if not (file := Path(file)).exists():
                raise ValueError(f'{file} does not exist.')
            if verbose:
                print(f'Reading: {file.name}.')
            with open(file, 'rb') as picklefile:
                return pickle.load(picklefile)
        else:
            raise TypeError(error_message)

    def get_summary(info):

        def extract_direct_value(info, attr):
            '''Handle info extraction'''
            try:
                value = getattr(info,attr)
                if value is None:
                    return np.nan
                else:
                    return value
            except AttributeError:
                return np.nan

        def extract_value(info, attr):
            '''Handle info extraction also for nested attributes'''
            # Convert to list
            if '.' in attr:
                attrs = attr.split('.')
            else:
                attrs = [attr]
            # Extract the nested value
            tmp = info
            for attr in attrs:
                tmp = extract_direct_value(tmp, attr)
                if tmp is np.nan:
                    break
            value = tmp
            return value
        
        # Estimates to extract
        summary = SimpleNamespace()
        summary.tic = extract_value(info, 'tic')
        summary.sector = extract_value(info, 'sector')
        summary.ra = extract_value(info, 'ra')
        summary.dec = extract_value(info, 'dec')
        summary.flux_contamination_fraction = extract_value(info, 'fit.fraction_contamination_ap')
        summary.background_change_fraction = extract_value(info, 'fit.fraction_bkg_change')
        summary.mag = extract_value(info, 'target.mag')
        summary.number_of_pc_used = extract_value(info, 'pca_used.npc')
        summary.aperture_mask_threshold = extract_value(info, 'aperture_threshold')
        mask = extract_value(info, 'masks.aperture')
        if mask is not np.nan:
            summary.aperture_mask_size = mask.sum()
        mask = extract_value(info, 'masks.background')
        if mask is not np.nan:
            summary.background_mask_size = mask.sum()
        summary.tag = extract_value(info, 'tag')
        lc = extract_value(info, 'lc_regressed_clean')
        if lc is not np.nan:
            span = lc.time.max() - lc.time.min()
            summary.time_span = span.value
            summary.time_points = lc.time.size
        
        return summary

    if isinstance(files,str):
        # If given a directory, get the list of files in it
        if Path(files).is_dir():
            inputdir = Path(files)
            # Get the filepaths and filenames of the files to process
            utils.contains_TIC(namePattern)
            globPattern = namePattern.replace('{TIC}', '*')
            filepaths = [file for file in inputdir.glob(globPattern)]
            filenames = [file.name for file in filepaths]
            filesTable = pd.DataFrame({'filename':filenames, 'filepath':filepaths})
            # Get TIC number from filename
            filesTable['tic'] = filesTable['filename'].apply(utils.return_TIC_2, args=(namePattern,))
            # Sort by TIC number
            filesTable.sort_values(by=['tic'], inplace=True)
            # Select the TIC number to process
            if not TICs == 'all':
                if isinstance(TICs,int):
                    filesTable = filesTable.query('tic == @TICs')  
                elif isinstance(TICs,(list,np.ndarray,pd.core.series.Series)):
                    filesTable = filesTable.query('tic in @TICs')  
                else:
                    raise ValueError(f'TICs must be "all", an integer or a list of integers. Got {TICs}')
            files = filesTable['filepath'].astype(str).values
        # If given a file, make it a list
        elif Path(files).is_file():
            files = [files]

    valid_types = (list, np.ndarray, pd.core.series.Series)
    if not isinstance(files,valid_types):
        raise TypeError(f'`files` must be a str, list, np.ndarray or pd.core.series.Series. Got {type(files)} instead.')

    summaries = []
    if progressbar:
        files = tqdm(files)
    for file in files:
        sectorsInfo = read_pickle_file(file)
        for sectorInfo in sectorsInfo:
            summaries.append(get_summary(sectorInfo).__dict__)
    
    if overwrite or not Path(csvname).exists():
        pd.DataFrame(summaries).to_csv(csvname, index=False)