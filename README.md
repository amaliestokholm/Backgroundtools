## Tools for Background
These are small tools written to work with [Background](https://github.com/EnricoCorsaro/Background).

Currently, they work if you place these scripts within your `Background` directory.

The scripts:
- `Run\ Background\ for\ multiple\ targets.ipynb` is a Jupyter notebook that allows the user to run multiple runs of Background.
It iterates over all stars with a `.txt` file in the directory `/data/` in Background.
Given a csv file with possible guess of numax, this script initialises the background priors.
It then outputs a shellscript, which the user can run in their terminal that fits all the stars with their set of priors.
This script can also produce the overview plots of each fit and make a summray directory and a summary file for a better overview.
- `backgroundevaluation.py` is a Python 3 script that makes it easier for the user to assess each fit.
In the terminal, run <code>python3 backgroundevaluation.py eval</code> for the evaluation mode.
This then loops over each star in `/data/` and makes a A4-page pdf of their power spectrum and convergence of the model parameters.
The user can then choose whether 
