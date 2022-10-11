# Tessutils
Pick a TIC number and obtain a reduced light curve for all TESS sectors available along with diagnostic plots about the reduction.

# Requirements
The present code runs with no issues under Python 3.10 with the following packages:

1. astropy                   	5.1 
2. astroquery                	0.4.6
3. joblib                    	1.1.0
4. lightkurve                	2.3.0
5. matplotlib                	3.5.1
6. numpy                     	1.21.5
7. pandas                    	1.4.3
8. peakutils                 	1.3.4
9. scipy                     	1.7.3
10. tqdm                      	4.64.0

You can cover all requirements by installing the following virtual environment using conda.

`conda create --name tessutils python=3.10 astropy=5.1 astroquery=0.4.6 joblib=1.1.0 lightkurve=2.3.0 matplotlib=3.5.1 numpy=1.21.5 pandas=1.4.3 peakutils=1.3.4 scipy=1.7.3 tqdm=4.64.0 --channel conda-forge`

Once created, you can activate the environment as by using the following command

`conda activate tessutils`

# Importing Tessutils
After downloading or cloning the repository (folder) tessutils and codes within it, we need to add its parent location to the environment variable used by Python to search for modules so that we can have access to the tessutils module regardless our location on our machine.

If the location of Tessutils on our machine is, for example,

`/Users/stefano/Documents/Python/myPackages/tessutils`

the location to add is then

`/Users/stefano/Documents/Python/myPackages`

One easy way to add such a location is by executing the following commands within our Python session 

$ import sys
$ sys.path.append('/Users/stefano/Documents/Python/myPackages')

We can now import Tessutils, for instance

$ import tessutils as tu

# Run example
Let us say we are interested in the star with TIC 374944608 from sector 7. 

First, we download the corresponding Target Pixel File (TPF)

$ import tessutils as tu
$ tu.reduction.download_tpf(374944608, sectors=7)

the TPF has been downloaded to a folder named `tpfs`. Second, we extract the light curve from the TPF

$ tu.reduction.extract_light_curve('tpfs/tic374944608_sec7.fits')

Finally, we create a diagnostic plot of the relevant processes involved during the light curve extraction

$ tu.plots.plot_diagnosis('processed/tic374944608_sec7_corrected.pickle')

# Further examples
See the Jupiter Notebook.

# Found an issue in the code?
Please then create an issue and include a minimal reproducible example of it.