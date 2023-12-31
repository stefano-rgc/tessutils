# Tessutils
Pick a TIC number and obtain a reduced light curve for all TESS sectors available along with diagnostic plots about the reduction.

# Requirements
The present code runs with no issues under Python 3.10 with the following packages:

1. [astropy](https://www.astropy.org) (5.1)
2. [astroquery](https://astroquery.readthedocs.io/en/latest/) (0.4.6)
3. [joblib](https://joblib.readthedocs.io/en/latest/#) (1.1.0)
4. [lightkurve](https://docs.lightkurve.org) (2.3.0)
5. [matplotlib](https://matplotlib.org/) (3.5.1)
6. [numpy](https://numpy.org) (1.21.5)
7. [pandas](https://pandas.pydata.org/) (1.4.3)
8. [peakutils](https://peakutils.readthedocs.io/en/latest/) (1.3.4)
9. [scipy](https://scipy.org/) (1.7.3)
10. [tqdm](https://github.com/tqdm/tqdm) (4.64.0)

If you will also run the [tutorial](https://github.com/stefano-rgc/tessutils/blob/main/tutorial/Tessutils_tutorial.ipynb) from the Jupyter notebook below

https://github.com/stefano-rgc/tessutils/blob/main/tutorial/Tessutils_tutorial.ipynb

then, the packages `IPython` and `ipympl` are also needed, plus I recommend opening the notebook using JupyterLab for a better experience.

You can cover all requirements by installing the following virtual environment using [conda](https://docs.conda.io/en/latest).

```
conda create --name tessutils python=3.10 astropy=5.1 astroquery=0.4.6 joblib=1.1.0 lightkurve=2.3.0 matplotlib=3.5.1 numpy=1.21.5 pandas=1.4.3 peakutils=1.3.4 scipy=1.7.3 tqdm=4.64.0 IPython ipympl --channel conda-forge
```

Once created, you can activate the environment as by using the following command

```
conda activate tessutils
```

⮕ Alternatively, you can use `pip` to directly install the package along with all its dependences. For it, go to the directory containing the `setup.py` file and run:

```
pip install .
```

If you wish so, you can later uninstall it with:

```
pip uninstall tessutils
``` 

# Importing Tessutils

If you used `pip` to install the package, then `tessutils` should be ready to import like any other Python module.

If you just downloaded or cloned the repository, then you need to add its location to the environment variable used by Python to search for modules, so that you can have access to the `tessutils` module regardless your location on your machine.

# Run example
Let us say we are within our Python session and interested in the star with TIC 374944608 from sector 7. 

First, we download the corresponding Target Pixel File (TPF)

```
import tessutils as tu
tu.reduction.download_tpf(374944608, sectors=7)
```

the TPF has been downloaded to a folder named `tpfs`. Second, we extract the light curve from the TPF

```
tu.reduction.extract_light_curve('tpfs/tic374944608_sec7.fits')
```

The reduced light curve, with systematics and outliers already removed, is a `lightkurve.lightcurve.TessLightCurve` object (from the module [lightkurve](https://docs.lightkurve.org)) stored as a pickle file along with information on the extraction process. Such a pickle file is stored by defaul in a folder named `processed` and can be accessed as follows

```
import pickle
picklefile = f'processed/tic374944608_sec7_corrected.pickle'
with open(picklefile, 'rb') as file:
    info = pickle.load(file)
lc = info.lc_regressed_clean
print(lc)
lc.plot()
```

Finally, we create a diagnostic plot of relevant processes involved during the light curve extraction

```
tu.plots.plot_diagnosis('processed/tic374944608_sec7_corrected.pickle')
```

# Further examples
See the Jupiter Notebook:

https://github.com/stefano-rgc/tessutils/blob/main/tutorial/Tessutils_tutorial.ipynb


# Found an issue in the code?
Create an [issue on GitHub](https://github.com/stefano-rgc/tessutils/issues) and include a minimal reproducible example of it.

# Reference paper
https://doi.org/10.1051/0004-6361/202141926
