# Built-in modules
from types import SimpleNamespace
import pickle
from pathlib import Path
# External modules
import numpy as np
import pandas as pd
import lightkurve as lk 
import matplotlib.pyplot as plt
import peakutils
import matplotlib as mpl
# Local modules
# import utils
import tessutils.utils as utils
# from reduction import normalize_lightCurve
from tessutils.reduction import normalize_lightCurve

def scalesymbols(mags,
                 min_mag,
                 max_mag,
                 scale=120):
    """
        Author: 
            Timothy
    
        Purpose:
            A simple routine to determine the scatter marker sizes, based on the TESS magnitudes
        
        Parameters:
            mags (numpy array of floats): the TESS magnitudes of the stars
            min_mag (float): the smallest magnitude to be considered for the scaling
            max_mag (float): the largest magnitude to be considered for the scaling
        Returns:
            sizes (numpy array of floats): the marker sizes
    """
    sizes = scale * (1.1*max_mag - mags) / (1.1*max_mag - min_mag)
    return sizes

def overplot_mask(ax,
                  mask,
                  ec='r',
                  fc='none',
                  lw=1.5,
                  alpha=1):
    """
        Author: 
            Timothy
    
        Purpose:
            (Over)plotting the mask (on a selected frame in the axis "ax").
    """
    for i in range(len(mask[:,0])):
        for j in range(len(mask[0,:])):
            if mask[i, j]:
                ax.add_patch(mpl.patches.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor=ec, facecolor=fc,linewidth=lw,alpha=alpha))

def plot_periodogram(ax,
                     pg,
                     snr=4,
                     fsize=7):
    """Plot periodogram on the given axes.

    Args:
        ax (matplotlib.axes.Axes):
            Axes to plot on.
        pg (lightkurve.periodogram.LombScarglePeriodogram):
            Periodogram to plot.
        snr (int, optional):
            Signal-noise ratio level of marked peaks in the periodogram.
            Defaults to 4.
        fsize (int, optional):
            Font size of the legend. Defaults to 7.

    Returns:
        None
    """

    def find_peaks(pg,
                   snr):
        """Find peaks in the periodogram.

        Args:
            pg (lightkurve.periodogram.LombScarglePeriodogram):
                Periodogram to be searched for peaks.
            snr (int, optional):
                Signal-noise ratio level of found peaks. Defaults to 4.

        Returns:
            background (lightkurve.periodogram.LombScarglePeriodogram):
                Background level of the periodogram.
            mask (numpy.ndarray):
                Boolean mask of elements with a signal-to-noise ratio higher
                than `snr`.
            ind (numpy.ndarray):
                Indices of `mask` corresponding to the peaks.
        """
        # Get the periodogram SNR by smoothing the power to obtain the background noise level
        spectrum, background = pg.flatten(method='logmedian', filter_width=0.3, return_trend=True)
        # Select SNR greater than `snr`
        mask = spectrum.power >= snr
        # Find the estimative period of the peaks
        ind = peakutils.indexes(spectrum.power[mask], thres=0.0, min_dist=1, thres_abs=True)
        return background, mask, ind

    background, mask, ind = find_peaks(pg,snr)

    x = pg.period.value
    y = pg.power.value
    ax.plot(x, y, zorder=2, color='k')
    # Plot peaks
    if len(ind) > 0:
        ax.scatter(x[mask][ind], y[mask][ind], marker='o', c='red', s=2, label=f'SNR > {snr}', rasterized=False, zorder=0, edgecolors='None', linewidth=0.1)
    # Plot background
    x = background.period.value
    y = background.power.value
    ax.fill_between(x, y, ls='-', lw=0.1, color='lime', label='Background', rasterized=False, zorder=0)
    # Axes limit
    if len(ind) > 0:
        scale_factor = 1.1
        xmin = x[mask][ind].min()
        xmax = x[mask][ind].max()
        xmax_minus_xmin = xmax - xmin
        xmax_plus_xmin = xmax + xmin
        xmax_new = 0.5*( scale_factor*xmax_minus_xmin + xmax_plus_xmin) 
        xmin_new = xmax_plus_xmin - xmax_new
    else:
        xmin_new = 0 
        xmax_new = 20
    ax.set_xlim(xmin_new,xmax_new)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=1, loc='best', fontsize=0.75*fsize, frameon=False, handletextpad=0.5, markerscale=1.5, handlelength=1)
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Amplitude\n(ppt)')

def overplot_sector_intervals(ax,
                              intervals):
    """Overplot intervals with alternating shades of gray on the given axes.

    Args:
        ax (matplotlib.axes.Axes):
            Axes to plot on.
        intervals (dict):
            Dictionary indicating the intervals to exclude. The keys are
            integers indicating the TESS sectors, and the values are lists of
            tuples, each tuple containing an initial and final time to exclude.
            Times must be given as a astropy.time.Time types.
            
    Returns:
        None
    """
    for i,(sector,interval) in enumerate(intervals.items()):
        interval = np.array([interval[0].value,interval[1].value]) # Remove astropy.time.Time element
        mean = np.mean(interval)
        _, y = ax.get_ylim()
        mean_axes_fraction, _ = ax.transAxes.inverted().transform((mean,0))
        ax.axvspan(*interval, facecolor='gray', alpha=0.30 if i%2==0 else 0.15, edgecolor='None')
        ax.annotate(f'{sector}', xy=(mean, y), ha='center', va='bottom', color='k')

def plot_normalized_stitched_light_curve_ppt(ax,
                                             lc,
                                             separate_intervals=True,
                                             separate_threshold=1.5,
                                             rasterized=False,
                                             intervals=None):
    """Plot normalized stitched light curve in ppt on the given axes.

    Args:
        ax (matplotlib.axes.Axes):
            Axes to plot on.
        lc ():

        separate_intervals (bool, optional):
            Whether to connect all cadences with a line on the plot. A value of
            True will separate cadences farther apart than the minimum cadence
            separation times `separate_threshold`. Defaults to True.
        separate_threshold (float, optional):
            Parameter that controls how far apart cadences must be to be plotted
            separately. See doc for `separate_intervals`. Defaults to 1.5.
        rasterized (bool, optional):
            Whether to rasterize the plot. Defaults to False.
        intervals (dict):
            If given, separate the cadences within the given intervals. Useful
            when plotting light curves with different cadence times.
            Dictionary indicating the intervals to exclude. The keys are
            integers indicating the TESS sectors, and the values are lists of
            tuples, each tuple containing an initial and final time to exclude.
            Times must be given as a astropy.time.Time types. Defaults to None.

    Returns:
        None
    """
    label = 'Stitched'
    color = 'k'
    if separate_intervals and intervals is not None:
        for sector,interval in intervals.items():
            mask = (lc.time >= interval[0]) & (lc.time <= interval[1])
            df = lc[mask].to_pandas().reset_index()
            df['dtime'] = df['time'].diff()
            ind = df.query('dtime > @separate_threshold*dtime.min()').index
            for i,_lc in enumerate(np.split(df,ind)):
                flux = _lc.flux.values
                time = _lc.time.values
                label = label if i == 0 else None
                ax.plot(time, flux, color=color, rasterized=rasterized, label=label)
    elif separate_intervals:
        df = lc.to_pandas().reset_index()
        df['dtime'] = df['time'].diff()
        ind = df.query('dtime > @separate_threshold*dtime.min()').index
        for i,lc in enumerate(np.split(df,ind)):
            flux = lc.flux.values
            time = lc.time.values
            label = label if i == 0 else None
            ax.plot(time, flux, color=color, rasterized=rasterized, label=label)
    else:
        flux = lc.flux.value
        time = lc.time.value
        ax.scatter(time, flux, s=2, marker='.', linewidths=0, color=color, rasterized=rasterized, label=label)
    ax.set_ylabel('Normalized\nflux (ppt)')
    ax.legend(ncol=1, loc='best', frameon=False)

def plot_sector(sectorInfo,
                figure,
                grid,
                verbose=True):
    """Plot a diagnostic figure for a given sector.

    Args:
        sectorInfo (_type_):
            SimpleNamespace object with the following attributes:
                * sectorInfo.tic
                * sectorInfo.sector
                * sectorInfo.ra
                * sectorInfo.dec
                * sectorInfo.headers
                * sectorInfo.fit
                * sectorInfo.neighbours_all
                * sectorInfo.neighbours_used
                * sectorInfo.target
                * sectorInfo.aperture_threshold
                * sectorInfo.pca_all
                * sectorInfo.pca_used
                * sectorInfo.centroids
                * sectorInfo.excluded_intervals
                * sectorInfo.lc_raw
                * sectorInfo.lc_raw_nonan
                * sectorInfo.lc_trend
                * sectorInfo.lc_regressed
                * sectorInfo.lc_regressed_notoutlier
                * sectorInfo.median_image
                * sectorInfo.masks
                * sectorInfo.tag
        figure (matplotlib.figure.Figure | None ):
            Figure to plot on. If None, a new figure will be created and returned.
        grid (matplotlib.gridspec.SubplotSpec):
            Gridspec to plot on.
        verbose (bool, optional):
            Whether use verbose output. Defaults to True.

    Returns:
        figure
    """

    def plot_images(fig,
                    grid,
                    info,
                    TitleFontSize):
        """Plot at most four images of diagnosis regarding the TPF and the light
        curve extraction on the given figure and grid.

        Args:
            figure (matplotlib.figure.Figure):
                Figure to plot on.
            grid (matplotlib.gridspec.SubplotSpec):
                Gridspec to plot on.
            info:
                SimpleNamespace object with the following attributes:
                    * info.tic
                    * info.sector
                    * info.ra
                    * info.dec
                    * info.headers
                    * info.fit
                    * info.neighbours_all
                    * info.neighbours_used
                    * info.target
                    * info.aperture_threshold
                    * info.pca_all
                    * info.pca_used
                    * info.centroids
                    * info.excluded_intervals
                    * info.lc_raw
                    * info.lc_raw_nonan
                    * info.lc_trend
                    * info.lc_regressed
                    * info.lc_regressed_notoutlier
                    * info.median_image
                    * info.masks
                    * info.tag
            TitleFontSize (float):
                Title font size.

        Returns:
            None
        """

        def plot_median_image(ax,
                              image,
                              TitleFontSize):
            """Plot median image of cadences on the given axes.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                image (numpy.ndarray):
                    Median image.
                TitleFontSize (float):
                    Title font size.

            Returns:
                None
            """
            ax.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
            ax.set_title(f'Median', size=TitleFontSize, pad=2)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        def plot_masks(ax,
                       image,
                       aperture,
                       background,
                       TitleFontSize):
            """Plot aperture and background masks on the given axes.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                image (numpy.ndarray):
                    Median image.
                aperture (numpy.ndarray):
                    2D boolean array indicating the aperture mask.
                background (numpy.ndarray):
                    2D boolean array indicating the background mask.
                TitleFontSize (float):
                    Title font size.
                    
            Returns:
                None
            """
            ax.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
            overplot_mask(ax,background,ec='w',lw=0.1, fc='w', alpha=0.3)
            overplot_mask(ax,background,ec='w',lw=0.1, fc='none', alpha=1.0)
            overplot_mask(ax,aperture,ec='r',lw=0.1, fc='r', alpha=0.3)
            overplot_mask(ax,aperture,ec='r',lw=0.1, fc='none', alpha=1.0)
            ax.set_title('Masks',size=TitleFontSize, pad=2)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            
        def plot_neighbourhood(ax,
                               image,
                               target,
                               neighbours,
                               mag_reference,
                               TitleFontSize):
            """Plot the target and its neighbouring stars on the given axes.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                image (numpy.ndarray):
                    Median image.
                target:
                    SimpleNamespace object with the following attributes:
                        * target.mag
                        * target.ra
                        * target.dec
                        * target.pix
                neighbours:
                    SimpleNamespace object with the following attributes:
                        * neighbours.mag
                        * neighbours.ra
                        * neighbours.dec
                        * neighbours.pix
                mag_reference (list[int]):
                    Reference magnitude to add to the legend.
                TitleFontSize (float):
                    Title font size.

            Returns:
                None
            """
            mags = np.r_[neighbours.mag, target.mag, mag_reference]
            ax.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
            ax.set_title('Neighbours',size=TitleFontSize, pad=2)
            sizes = scalesymbols(target.mag,np.amin(mags), np.amax(mags), scale=15.)
            ax.scatter(target.pix[:,0],target.pix[:,1],s=sizes,c='r',edgecolors='k', linewidth=0.2, zorder=5, label=f'{target.mag:.1f}')
            if neighbours.pix.size > 0:
                sizes = scalesymbols(neighbours.mag,np.amin(mags), np.amax(mags), scale=15.)
                ax.scatter(neighbours.pix[:,0],neighbours.pix[:,1],s=sizes,c='w',edgecolors='k', linewidth=0.2, zorder=5)
                for size in mag_reference:
                    sizes = scalesymbols(size,np.amin(mags), np.amax(mags), scale=15.)
                    ax.scatter(-1, -1,s=sizes,c='w',edgecolors='k', linewidth=0.2, label=f'{size}')
            ax.set_xlim(0,image.shape[0]-1)
            ax.set_ylim(0,image.shape[1]-1)
            ax.legend(ncol=4, loc=(-1.0,-0.3), frameon=False, handletextpad=0.1, columnspacing=0.)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        def plot_fit(ax,
                     image,
                     TitleFontSize):
            """Plot the fit of the median image on the given axes.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                image (numpy.ndarray):
                    Fit of the median image.
                TitleFontSize (float):
                    Title font size.

            Returns:
                None
            """
            ax.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_title('Fit',size=TitleFontSize, pad=2)

        # Create axes for the images
        main2Rows2Cols = grid.subgridspec(2, 2, hspace=0, wspace=0)
        ax1 = fig.add_subplot(main2Rows2Cols[0,0])
        image = info.median_image
        plot_median_image(ax1, image, TitleFontSize)

        # Stop here if no aperture mask is available nor neighbor stars used       
        if info.masks.aperture is None and info.neighbours_used is None:
            return False

        # Plot the aperture and background masks
        if info.masks.aperture is not None:
            aperture_mask = info.masks.aperture.astype(bool)
            backgroung_mask  = info.masks.background.astype(bool)
            ax2 = fig.add_subplot(main2Rows2Cols[1,0])
            plot_masks(ax2,image,aperture_mask,backgroung_mask,TitleFontSize)    
                        
        # Stop here if no neighbor stars used       
        if info.neighbours_used is None:
            return False
        
        # Plot neighbour stars
        mag_reference = [10, 12, 14]
        ax3 = fig.add_subplot(main2Rows2Cols[1,1])
        plot_neighbourhood(ax3,image,info.target,info.neighbours_used,mag_reference,TitleFontSize)

        # Stop here if no fit is available
        if info.fit is None:
            return False
        
        # Plot fit
        ax4 = fig.add_subplot(main2Rows2Cols[0,1])
        image = info.fit.fitted_image
        plot_fit(ax4,image,TitleFontSize)

        # Return True if the plot reaches a satisfactory fitted image
        return True

    def plot_timeSeries(fig,grid,info,TitleFontSize):
        """Plot at most three diagnosis plots regarding the light curve
        extraction on the given figure and grid.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to plot on.
            grid (matplotlib.gridspec.SubplotSpec):
                Gridspec to plot on.
            info:
                SimpleNamespace object with the following attributes:
                    * info.tic
                    * info.sector
                    * info.ra
                    * info.dec
                    * info.headers
                    * info.fit
                    * info.neighbours_all
                    * info.neighbours_used
                    * info.target
                    * info.aperture_threshold
                    * info.pca_all
                    * info.pca_used
                    * info.centroids
                    * info.excluded_intervals
                    * info.lc_raw
                    * info.lc_raw_nonan
                    * info.lc_trend
                    * info.lc_regressed
                    * info.lc_regressed_notoutlier
                    * info.median_image
                    * info.masks
                    * info.tag
            TitleFontSize (float):
                Title font size.

        Returns:
            None
        """
        def plot_centroids(ax,
                           centroids,
                           separate_intervals=True,
                           separate_threshold=1.5,
                           rasterized=False):
            """Plot the centroid of the target star over time on the given axis.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                centroids:
                    SimpleNamespace object with the following attributes:
                        * centroids.col
                        * centroids.row
                        * centroids.sqrt_col2_row2
                        * centroids.time
                separate_intervals (bool, optional):
                    Whether to connect all cadences with a line on the plot. A value of
                    True will separate cadences farther apart than the minimum cadence
                    separation times `separate_threshold`. Defaults to True.
                separate_threshold (float, optional):
                    Parameter that controls how far apart cadences must be to be plotted
                    separately. See doc for `separate_intervals`. Defaults to 1.5.
                rasterized (bool, optional):
                    Whether to rasterize the plot. Defaults to False.

            Returns:
                None
            """
            label = 'Centroid'
            color = 'r'
            centroid = centroids.sqrt_col2_row2
            time = centroids.time
            if separate_intervals:
                df = pd.DataFrame({'centroid':centroid, 'time':time})
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for i,centroids in enumerate(np.split(df,ind)):
                    centroid = centroids.centroid.values
                    time = centroids.time.values
                    label = label if i == 0 else None
                    ax.plot(time, centroid, color=color, rasterized=rasterized, label=label)
            else:
                ax.plot(time, centroid , color=color, rasterized=rasterized, label=label)
            ax.set_ylabel('Pix')
            ax.label_outer()
            ax.legend(ncol=1, loc='best', frameon=False)

        def plot_raw_light_curve_and_trend(ax,
                                           lc,
                                           trend,
                                           separate_intervals=True,
                                           separate_threshold=1.5,
                                           rasterized=False):
            """Plot the raw light curve and trend on the given axis.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                lc (lightkurve.lightcurve.TessLightCurve):
                    Light curve to plot.
                trend (lightkurve.lightcurve.TessLightCurve):
                    Trend to plot.
                separate_intervals (bool, optional):
                    Whether to connect all cadences with a line on the plot. A value of
                    True will separate cadences farther apart than the minimum cadence
                    separation times `separate_threshold`. Defaults to True.
                separate_threshold (float, optional):
                    Parameter that controls how far apart cadences must be to be plotted
                    separately. See doc for `separate_intervals`. Defaults to 1.5.
                rasterized (bool, optional):
                    Whether to rasterize the plot. Defaults to False.

            Returns:
                None
            """
            color_lc = 'k'
            label_lc = 'Raw'
            color_trend = 'dodgerblue'
            label_trend = 'Trend'
            if separate_intervals:
                df = lc.to_pandas().reset_index()
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for i,lc in enumerate(np.split(df,ind)):
                    flux = lc.flux.values
                    time = lc.time.values
                    label = label_lc if i == 0 else None
                    ax.plot(time, flux, color=color_lc, rasterized=rasterized, label=label)
                df = trend.to_pandas().reset_index()
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for i,trend in enumerate(np.split(df,ind)):
                    flux = trend.flux.values
                    time = trend.time.values
                    label = label_trend if i == 0 else None
                    ax.plot(time, flux, color=color_trend, rasterized=rasterized, label=label)
            else:
                lcTime = lc.time.value
                lcFlux = lc.flux.value
                trendTime = trend.time.value
                trendFlux = trend.flux.value
                ax.plot(lcTime,lcFlux, color=color_lc, rasterized=rasterized, zorder=3, label=label_lc)
                ax.scatter(trendTime,trendFlux, color=color_trend,marker='*', rasterized=rasterized, s=1, zorder=4, linewidths=0, label=label_trend)
            ax.set_ylabel('e$^{-}$/s')
            ax.legend(ncol=1, loc='best', frameon=False)
            ax.label_outer()
            ax.set_facecolor('whitesmoke')

        def plot_detrended_light_curve(ax,
                                       lc,
                                       separate_intervals=True,
                                       separate_threshold=1.5,
                                       rasterized=False):
            """Plot the detrended light curve on the given axis.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                lc (lightkurve.lightcurve.TessLightCurve):
                    Light curve to plot.
                separate_intervals (bool, optional):
                    Whether to connect all cadences with a line on the plot. A value of
                    True will separate cadences farther apart than the minimum cadence
                    separation times `separate_threshold`. Defaults to True.
                separate_threshold (float, optional):
                    Parameter that controls how far apart cadences must be to be plotted
                    separately. See doc for `separate_intervals`. Defaults to 1.5.
                rasterized (bool, optional):
                    Whether to rasterize the plot. Defaults to False.

            Returns:
                None
            """
            label = 'Detrended'
            color = 'k'
            if separate_intervals:
                df = lc.to_pandas().reset_index()
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for i,_lc in enumerate(np.split(df,ind)):
                    flux = _lc.flux.values
                    time = _lc.time.values
                    label = label if i == 0 else None
                    ax.plot(time, flux, color=color, rasterized=rasterized, label=label)
            else:
                flux = lc.flux.value
                time = lc.time.value
                ax.plot(time, flux, color=color, rasterized=rasterized, label=label)
            
            # Set ylabel
            ylabel = lc.flux.unit.to_string()
            if 'electron' in ylabel:
                 ylabel = ylabel.replace('electron','e$^{-}$')
            ax.set_ylabel(ylabel)
            # Set xlabel
            xlabel = utils.parse_lc_time_units(lc)
            ax.set_xlabel(xlabel)
            ax.label_outer()
            ax.legend(ncol=1, loc='best', frameon=False)

        def overplot_excluded_intervals(ax,
                                        intervals,
                                        label=None):
            """Overplot excluded intervals region on the given axis.

            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot on.
                intervals (dict):
                    Dictionary indicating the intervals to exclude. The keys are
                    integers indicating the TESS sectors, and the values are lists of
                    tuples, each tuple containing an initial and final time to exclude.
                    Times must be given as a astropy.time.Time types.
                label (_type_, optional):
                    Label for the legend. Defaults to None.

            Returns:
                None
            """
            if intervals is not None:
                for i,interval in enumerate(intervals):
                    _label = label if i == 0 else None
                    ax.axvspan(*interval.value, color='yellow', alpha=0.3, label=_label)
                if label is not None and len(intervals)>0:
                    ax.legend(ncol=1, loc='best', frameon=False)

        # Create the 3 stacked axes in column 1
        main3Rows = grid.subgridspec(3, 1, hspace=0)
        ax0 = fig.add_subplot(main3Rows[0])
        ax1 = fig.add_subplot(main3Rows[1], sharex=ax0)
        ax2 = fig.add_subplot(main3Rows[2], sharex=ax0)

        for ax in [ax0,ax1,ax2]:
            label = 'Excluded' if ax == ax2 else None # Only label the last axis
            overplot_excluded_intervals(ax,info.excluded_intervals,label=label)

        plot_centroids(ax0,info.centroids)

        # If not trend is available, stop here and display the error message
        if info.lc_trend is None:
            text = '\n'.join([t for t in utils.chunks(info.tag,47)]  )
            ax2.text(0.5,0.5, text, transform=ax2.transAxes, fontsize=1.1*TitleFontSize, ha='center', va='center', color='green', wrap=True)
            ax.legend(ncol=1, loc='best', frameon=False)
            ax2.label_outer()
            return False

        plot_raw_light_curve_and_trend(ax1,info.lc_raw,info.lc_trend)
        
        plot_detrended_light_curve(ax2,info.lc_regressed_notoutlier)

        return True

    def plot_principalComponents(ax,
                                 info,
                                 separate_intervals=False,
                                 separate_threshold=1.5,
                                 rasterized=False):
        """Plot the principal components on the given axis. The first principal
        component is plotted at the bottom and later ones are plotted in order
        above it. Used principal components are plotted with colors while unused
        ones are plotted in black.

        Args:
            ax (matplotlib.axes.Axes):
                Axes to plot on.
            info: 
                SimpleNamespace object with the following attributes:
                    * info.pca_used
                    * info.lc_regressed
                    * info.pca_all
            separate_intervals (bool, optional):
                Whether to connect all cadences with a line on the plot. A value of
                True will separate cadences farther apart than the minimum cadence
                separation times `separate_threshold`. Defaults to True.
            separate_threshold (float, optional):
                Parameter that controls how far apart cadences must be to be plotted
                separately. See doc for `separate_intervals`. Defaults to 1.5.
            rasterized (bool, optional):
                Whether to rasterize the plot. Defaults to False.

        Returns:
            None
        """
        # Get the default color list and doble it
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [*colors,*colors]

        # Number of used principal components minus the constant one
        n = info.pca_used.npc 
        time = info.lc_regressed.lc.time.value
        # Iterate over the no constant principal components
        for i,pc in enumerate(info.pca_used.pc[:-1]):
            offset = i*0.2
            color = colors[i]
            if separate_intervals:
                df = pd.DataFrame({'time':time,'pc':pc})
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for j,pc in enumerate(np.split(df,ind)):
                    ax.plot(pc.time.values, pc.pc.values+offset, color=color, rasterized=rasterized)
            else:
                ax.scatter(time,pc+offset, marker='.', rasterized=rasterized, s=1, zorder=3, linewidths=0)
            # ax.scatter(time,pc+offset, marker='.', rasterized=True, s=1, zorder=3, linewidths=0)
        for pc in info.pca_all.pc[:-1][n:]:
            i+=1
            offset = i*0.2
            color = 'k'
            if separate_intervals:
                df = pd.DataFrame({'time':time,'pc':pc})
                df['dtime'] = df['time'].diff()
                ind = df.query('dtime > @separate_threshold*dtime.min()').index
                for j,pc in enumerate(np.split(df,ind)):
                    ax.plot(pc.time.values, pc.pc.values+offset, color=color, rasterized=rasterized)
            else:
                ax.scatter(time,pc+offset, marker='.', color=color, rasterized=rasterized, s=1, zorder=2, linewidths=0)
        ax.axes.get_yaxis().set_visible(False)
        # Set xlabel
        xlabel = utils.parse_lc_time_units(info.lc_regressed.lc, short=True)
        ax.set_xlabel(xlabel)
        ax.label_outer()
        ax.set_facecolor('whitesmoke')

    def set_mpl_style():
        """Set the matplotlib style.

        Returns:
            None
        """
        fsize=7
        plt.rcParams['axes.labelsize'] = fsize
        plt.rcParams['axes.titlesize'] = fsize
        plt.rcParams['legend.fontsize'] = fsize/1.5
        plt.rcParams['xtick.major.size'] = fsize/7
        plt.rcParams['xtick.major.width'] = fsize/7/10
        plt.rcParams['xtick.minor.size'] = 0
        plt.rcParams['xtick.minor.width'] = fsize/7/10
        plt.rcParams['ytick.major.size'] = fsize/7
        plt.rcParams['ytick.major.width'] = fsize/7/10
        plt.rcParams['ytick.minor.size'] = 0
        plt.rcParams['ytick.minor.width'] = fsize/7/10
        plt.rcParams['axes.linewidth'] = fsize/7/10
        plt.rcParams['lines.linewidth'] = fsize/7/10
        plt.rcParams['xtick.labelsize'] = fsize*0.75
        plt.rcParams['ytick.labelsize'] = fsize*0.75

    if figure is None:
        width = 8.27
        height = 2.2
        figure = plt.figure(figsize=(width, height), constrained_layout=False, dpi=300)
    main3Cols = grid.subgridspec(1, 3, width_ratios=[5,1,1], wspace=0)
    # Set plotting parameters
    fsize=7
    set_mpl_style()
    if verbose:
        print(f'Ploting TIC {sectorInfo.tic}, sector {sectorInfo.sector}.')
    # Display the TIC number on the plot
    ax = figure.add_subplot(main3Cols[0])
    ax.axis('off')
    ax.set_title(f'TIC {sectorInfo.tic}')
    # Add title to the principal component axes
    ax = figure.add_subplot(main3Cols[1])
    ax.axis('off')
    ax.set_title('Principal components')
    # Display the sector number on the plot
    ax = figure.add_subplot(main3Cols[2])
    ax.axis('off')
    ax.set_title(f'Sector {sectorInfo.sector}', pad=2, weight='bold')
    # Plot TESS images
    ok = plot_images(figure,main3Cols[2],sectorInfo,fsize*0.75)
    # If not all images
    if not ok:
        ax = figure.add_subplot(main3Cols[0])
        text = sectorInfo.tag
        text = '\n'.join([t for t in utils.chunks(text,47)]  )
        ax.text(0.5,0.5, text, transform=ax.transAxes, fontsize=1.1*fsize, ha='center', va='center', color='green', wrap=True)
    else:
        # Plot time series
        ok = plot_timeSeries(figure,main3Cols[0],sectorInfo,fsize)
    if ok:
        ax = figure.add_subplot(main3Cols[1])
        plot_principalComponents(ax, sectorInfo)
    return figure

def plot_diagnosis(sectorInfo,
                   verbose=True,
                   pdfname=None,
                   pg_snr=4):
    """
    Purpose:
        Generate a PDF file with the diagnosis plots obtained from the output of
        `extract_light_curve` (one TESS sector) or `group_lcs` (a list of TESS sectors).

    Args:
        sectorInfo (str | pathlib.Path | SimpleNamespace | list[SimpleNamespace]):
            The output of `extract_light_curve` or `group_lcs`. If str or
            pathlib.Path, it must be the path to the pickle output file. If a
            SimpleNamespace, it must be the direct output of `extract_light_curve`,
            i.e. the content of the pickle file. If a list of SimpleNamespace, 
            it must be the direct output of `group_lcs`, i.e. the content of the
            pickle file.
        verbose (bool, optional):
            Verbose output. Defaults to True.
        pdfname (str | None, optional):
            Filename of the PDF file to be created. If None, then the filename
            will be similar to 'diagnostic_plot_TIC.pdf'. Defaults to None.
        pg_snr (int, optional):
            Signal-noise ratio level of marked peaks in the periodogram.
            Defaults to 4.

    Returns:
        None
    """
    # Read pickle file if needed
    if isinstance(sectorInfo,(str,Path)):
        if not (sectorInfo := Path(sectorInfo)).exists():
            raise ValueError(f'{sectorInfo} does not exist.')
        if verbose:
            print(f'Reading: {sectorInfo.name}')
        with open(sectorInfo, 'rb') as picklefile:
            sectorInfo = pickle.load(picklefile)
    # Validate input
    error_message = '`sectorInfo` must be a str or pathlib.Path pointing to the pickle file created by `extract_light_curve` or `group_lcs`. Altervatively, it can also be the SimpleNamespace, or a list of those, cointained in the pickle file.'
    if isinstance(sectorInfo,SimpleNamespace):
        pdfname = pdfname if isinstance(pdfname,str) else f'diagnostic_plot_TIC{sectorInfo.tic}_sector{sectorInfo.sector}.pdf'
        sectorsInfo = [sectorInfo] # Make into a list
    elif isinstance(sectorInfo,list):
        if all([isinstance(info,SimpleNamespace) for info in sectorInfo]):
            pdfname = pdfname if isinstance(pdfname,str) else f'diagnostic_plot_TIC{sectorInfo[0].tic}.pdf'
            sectorsInfo = sectorInfo
        else:
            raise TypeError(error_message)
    else:
        raise TypeError(error_message)
    nsectors = len(sectorsInfo) # Number of sectors to plot
    # Create figure
    width = 8.27
    height = 2.5 + 2.5*nsectors
    figure = plt.figure(figsize=(width, height), constrained_layout=False, dpi=300)
    height_ratios = np.append(np.repeat(2,nsectors), [1,1]).tolist()
    main3Rows = figure.add_gridspec(2+nsectors, 1, height_ratios=height_ratios, hspace=0.5)
    # Initialize list to collect light curves and intervals from each sector
    lcs = []
    sector_intervals = {}
    # One sector at a time
    for i,sectorInfo in enumerate(sectorsInfo):
        figure = plot_sector(sectorInfo, figure, main3Rows[i], verbose=verbose)
        # Collect only if the light curve extraction was successful
        if sectorInfo.tag == 'OK':
            lcs.append(sectorInfo.lc_regressed_notoutlier) # Detrended light curve without outliers
            sector_intervals[sectorInfo.sector] = utils.minmax(sectorInfo.lc_raw.time) # Time interval
    if len(lcs) >= 1:
        # Print status
        if verbose:
            print(f"Generating plot of stitched light curve and periodogram.")
        # Sticht the light curve
        lc = lk.LightCurveCollection(lcs).stitch(corrector_func=normalize_lightCurve)        
        # Make it ppt
        lc *= 1000
        # Plot the stitched light curve
        ax = figure.add_subplot(main3Rows[-2])
        plot_normalized_stitched_light_curve_ppt(ax,lc,separate_intervals=True,intervals=sector_intervals)
        overplot_sector_intervals(ax, sector_intervals)
        # Generate the Lomb-Scarglet periodogram
        pg = lc.to_periodogram()
        # Plot the periodogram
        ax = figure.add_subplot(main3Rows[-1])
        plot_periodogram(ax, pg,snr=pg_snr)
    if verbose:
        print(f"Saving PDF as {pdfname}.")
    figure.savefig(pdfname, bbox_inches='tight', dpi=300)
    return