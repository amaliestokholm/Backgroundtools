## Tools for Background
These are small tools written to work with [Background](https://github.com/EnricoCorsaro/Background).
This is very much work in progress, but I will upload them here for version control.

Currently, these scripts should work if you place them within your `Background` directory.

### The scripts
#### `Run\ Background\ for\ multiple\ targets.ipynb`

This is a Jupyter notebook that allows the user to run multiple runs of Background.

It iterates over all stars with a `.txt` file in the directory `/data/` in Background.
Given a csv file with possible guess of numax, this script initialises the background priors.
It then outputs a shellscript, which the user can run in their terminal and which fits all the stars with their computed set of priors.

This script can also produce overview plots of each fit and make a summary outputs for a better overview of all the fits in the given run.

#### `backgroundevaluation.py`

Thisis a Python 3 script that makes it easier for the user to assess each fit in a run and to make changes for refitting stars with a bad fit.

It loops over each star in `/data/` and makes a single A4-page pdf with their power spectrum and all convergence plots of the model parameters.
On a star-to-star basis, the plot is produced and showed to the user and the user can now choose whether to: (a) keep the fit, (b) change the model or model parameters of a fit, or (c) discard the star from the sample alltogether.

This script also produces a shellscript that makes it possible to rerun the stars the user wishes to rerun with a new set of parameters.


### Tutorial for `backgroundevaluation.py`
This script has two modes: `eval` and `retry`.

In the terminal, run

	python3 backgroundevaluation.py eval

for the evaluation mode.

The script should now ask you questions to which you can reply in the terminal by writing e.g. `y` and press enter.

The first questions revolves around specifying:
- an `idstr`. This is just en identifier sting used for the naming of files. This could be `test` or `kepler` or whatever you fancy.
- a `run`. This is the ID number of your Background run. If you have only run one set of models, this is `0`.

The script creates a set of directories and files and now the evaluation can begin.
All decisions, parameters, and evaluations are written to a logfile within the `evaluation` directory.

If you wish to interrupt the evaluation, you can do so by a keyboard interrupt.

If you reenter the same `idstr` and `run` the next time you run the program, you get the option to resume or delete everything and start all over.



The script also has a different mode, namely `retry`.
If you run

	python3 backgroundevaluation.py retry

this will then ask you to specify `idstr` and `run` just as the evaluation mode.
However, this mode will make a shellscript for the stars marked to be refitting in the logfile in the next run.
When the shell script is made, the user can make it executable by running

	chmod +x runBackgroundXX.sh

where XX is the run number, e.g. `01`

The user can then navigate to their ./build/` directory and run the command

	./runBackgroundXX.sh

and the fits will now run one after the other in the background.
If an error occurs or the fit fails, the name of the star will be added to a txt-file.

The `eval` mode also ends by calling the `retry` mode.
