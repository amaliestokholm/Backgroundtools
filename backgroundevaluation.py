# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate background fits for all stars in ./data/
# amalie
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import sys
import csv
import shutil
import numpy as np
import pandas as pd
from datetime import date, datetime
#import subprocess
import webbrowser
import argparse

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/1868714/
# how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth/22331852
from distutils.dir_util import copy_tree

sys.path.insert(0, "./results/python")
import background as bg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialise
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fieldnames = ['date', 'star', 'decision', 'notes', 'model', 'params']

backgroundmodels = ['FlatNoGaussian', 'Flat', 'Original',
                    'OneHarveyNoGaussian', 'TwoHarveyNoGaussian',
                    'ThreeHarveyNoGaussian',
                    'OneHarvey', 'TwoHarvey', 'ThreeHarvey',
                    'OneHarveyColor', 'TwoHarveyColor', 'ThreeHarveyColor']

# Define acceptable user input
goodinput = ['yes', 'y', 'Y', 'Yes']
noinput = ['no', 'n', 'N', 'No']
discardinput = ['discard', 'd', 'D']
retryinput = ['retry', 'r', 'R']
manualmode = ['manual', 'm', 'M']
numaxmode = ['numax', 'n', 'N']
fullinput = goodinput + discardinput + retryinput
yninput = goodinput + noinput



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sanitised_input(type_=None, min_=None, max_=None, range_=None):
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    uo = None
    while uo is None:
        ui = input()
        if ui == '':
            print('Please input something')
            continue
        elif type_ is not None:
            try:
                uo = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        elif max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
            continue
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
            continue
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))

            print('Not in range', range_)
            print('Please try again')
            continue
        else:
            uo = ui
    return uo


def run_numaxmode(star, model_name, newrun):
    prefix = re.split('(\d+)', star)[0]
    star_id = re.split('(\d+)', star)[1]

    print('What is your estimate of numax?')
    usernumax = sanitised_input(type_=float)

    bg.set_background_priors(
        catalog_id=prefix,
        star_id=star_id,
        numax=usernumax,
        model_name=model_name,
        dir_flag=int(newrun))
    print('set_background_priors has made a new hyperparameter file of run',
          newrun)


def make_pdffigure(pdffigure, datafile, computationfile, summaryfile,
                   paramfiles, resultdir, runresultdir):
    with PdfPages(pdffigure) as pdf:
        fig = plt.figure(figsize=(8.27, 11.7)) # A4 format-ish
        ncols = 3
        gs = fig.add_gridspec(6, ncols)
        ax1 = fig.add_subplot(gs[0:2, :])

        # Here I adjusted code from `background_plot`
        freq, psd = np.loadtxt(datafile, unpack=True)
        config = np.loadtxt(computationfile, unpack=True, dtype=str)
        model_name = config[-2]

        # Plot best-fitting model or initial guess
        if os.path.isfile(summaryfile):
            print('Success!')
            success = True
            params = np.loadtxt(summaryfile, usecols=(1,))
            numax = params[-2]
            bgf = bg.background_function(params,
                                         freq,
                                         model_name,
                                         star_dir=resultdir + '/')
            b1, b2, h_long, h_gran1, h_gran2, h_gran_original, g, w, h_color = bgf
        else:
            print('Run failed')
            success = False
            params = None
            numax = 50

            dnu = 0.267 * numax ** 0.760
            freqbin = freq[1] - freq[0]
            width = dnu / freqbin
            win_len= int(width)
            if win_len % 2 == 0:
                win_len += 1
            psd_smth = bg.smooth(psd, window_len=win_len, window='flat')

            ax1.loglog(freq, psd, color='grey')
            ax1.set_xlim(np.min(freq), np.max(freq))
            ax1.set_ylim(np.min(psd)*0.1, np.max(psd))
            ax1.set_xlabel(r'Frequency [$\mu$Hz]')
            ax1.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]')
            ax1.tick_params(width=1.5, length=8, top=True, right=True)
            ax1.tick_params(which='minor', length=6, top=True, right=True)

            ax1.plot(freq, psd_smth, 'k', lw=2)
            if params is not None:
                ax1.plot(freq, g, 'm-.', lw=2)
                ax1.plot(freq, h_color, 'y-.', lw=2)
                ax1.plot(freq, h_long, 'b-.', lw=2)
                ax1.plot(freq, h_gran1, 'b-.', lw=2)
                ax1.plot(freq, h_gran2, 'b-.', lw=2)
                ax1.plot(freq, h_gran_original, 'b-.', lw=2)
                ax1.plot(freq, w, 'y-.', lw=2)
                ax1.plot(freq, b1, 'r-', lw=3)
                ax1.plot(freq, b2, 'g--', lw=2)

            # Here I just copied the code from `parameter_evolution` but into a subplot
            for i, pf in enumerate(paramfiles):
                ax2 = fig.add_subplot(gs[2+ (i // ncols), i % ncols])
                sampling = np.loadtxt(os.path.join(runresultdir, pf), unpack=True)
                ax2.set_xlim(0, sampling.size)
                ax2.set_ylim(np.min(sampling),np.max(sampling))
                ax2.set_xlabel(r'Nested iteration')
                ax2.set_ylabel(pf.split('_')[1].split('.')[0])
                ax2.plot(np.arange(sampling.size),sampling,'k', lw=1)
            plt.tight_layout()
            pdf.savefig(fig)
    return success, params, model_name


def make_retryshellscript(idstr=None, run=None, newrun=None):
    # Read from logfile in order to properly handle resumed evaluations
    if idstr is None or run is None:
        print('Welcome to Background Evaluation')
        print('Here we will make a shell script for rerunning failed fits')
        print('Please define an id-string')
        idstr = sanitised_input(type_=str)

        print('Please define the run-string (e.g. 00 or 01)')
        run = sanitised_input(type_=int)
        run = str(run)
        if len(run) < 2:
            run = '0' + run

    if newrun is None:
        newrun = str(int(run)+1)
        if len(newrun) < 2:
            newrun = '0' + newrun

    retryshellscript = 'runBackground' + newrun + '.sh'

    today = date.today().strftime('%Y%m%d')
    evaldir = './evaluation/'
    logfile = os.path.join(evaldir, today + '_' + idstr + 'run_' + run + '.csv')
    errorfile = os.path.join(evaldir, 'errors_' + run + '.txt')
    if not os.path.isfile(logfile):
        print('Log file does not exist, exiting')
        return

    log = pd.read_csv(logfile, delimiter='\t')
    retrymask = ([d in retryinput for d in log['decision']])

    if np.sum(retrymask) > 0:
        with open(retryshellscript, "w") as f:
            print("#!/bin/bash", file=f)
            print("""
            f() {
                    if ! ./background "$@" ; then
                            echo "[`date`] $@" >> %s
                    fi
            }""" % errorfile, file=f)
            print('', file=f)
            for star, model in zip(log[retrymask]['star'], log[retrymask]['model']):
                prefix = re.split('(\d+)', star)[0]
                star_id = re.split('(\d+)', star)[1]
                print(f"f {prefix} {star} {newrun} {model} background_hyperParameters 0.0 0",
                      file=f)

        print('\n')
        print('A new shell script for rerunning fits has been made and is called', retryshellscript)
        print('In the terminal, make this file executable by running')
        print(f'chmod +x {retryshellscript}')
        print('Navigate to your `./build/` directory and run the command')
        print(f'../{retryshellscript}')
        print('\n')
        print('All the fits will now run one after the other.')
        print('If an error happens and a fit goes not go as intended,')
        print('the fit will be appended to the file', errorfile)
    else:
        print('No stars marked `retry` found in logfile')


def evaluate():
    print('Welcome to Background Evaluation')
    print('Please define an id-string')
    idstr = sanitised_input(type_=str)

    print('Please define the run-string (e.g. 00 or 01)')
    run = sanitised_input(type_=int)
    run = str(run)
    if len(run) < 2:
        run = '0' + run

    today = date.today().strftime('%Y%m%d')
    datadir = './data/'

    evaldir = './evaluation/'
    runevaldir = os.path.join(evaldir, idstr + '_run' + run)
    goodevaldir = os.path.join(runevaldir, 'goodfits')
    discardfile = os.path.join(runevaldir, 'discardstars.txt')
    logfile = os.path.join(evaldir, today + '_' + idstr + 'run_' + run + '.csv')
    errorfile = os.path.join(evaldir, 'errors_' + run + '.txt')

    newrun = str(int(run)+1)
    if len(newrun) < 2:
        newrun = '0' + newrun

    if not os.path.exists(evaldir):
        print('New evaluation directory created')
        os.mkdir(evaldir)

    if not os.path.exists(runevaldir):
        print('New run evaluation directories created')
        os.mkdir(runevaldir)
        os.mkdir(goodevaldir)
        resume = False
    else:
        print('Old evaluation folder for this run detected')
        print('Do you want to resume evaluation? y/n')
        userresume = sanitised_input(range_=yninput)
        if userresume in goodinput:
            resume = True
        else:
            print('This will overwrite all earlier evaluations')
            print('Are you sure? y/n')
            usersure = sanitised_input(range_=yninput)
            if usersure in goodinput:
                resume = False
                shutil.rmtree(goodevaldir, ignore_errors=True)
                os.mkdir(goodevaldir)
                if os.path.isfile(discardfile):
                    os.remove(discardfile)
            else:
                print('Evaluation is resumed')
                resume = True

    # Find all stars in datadir
    starlist = []
    for star in os.listdir(datadir):
        if star.endswith('.txt'):
            starlist.append(star.split('.')[0])

    assert len(starlist) > 0, 'No stars found - check ./data/'

    if resume:
        mode = 'a'
        if os.path.isfile(discardfile):
            discardstarlist = np.loadtxt(discardfile, dtype=str, unpack=True)

            for d in np.atleast_1d(discardstarlist):
                print('Star is discarded:', d)
                assert d in starlist
                starlist.remove(d)
        for g in os.listdir(goodevaldir):
            assert g in starlist
            print('Star is already evaluated:', g)
            starlist.remove(g)
    else:
        mode = 'w'
        print('Iterating over all stars in', datadir)

    # Make log
    with open(logfile, mode, newline='') as log:
        writer = csv.DictWriter(log, fieldnames=fieldnames, delimiter='\t')
        if mode == 'w':
            writer.writeheader()

        for star in starlist:
            print('')
            print('Evaluating', star)
            # Navigate to directory
            datafile = os.path.join('./data/' + star + '.txt')
            resultdir = os.path.join('./results/' + star)
            runresultdir = os.path.join(resultdir, run)
            summaryfile = os.path.join(runresultdir,
                                       'background_parameterSummary.txt')
            computationfile = os.path.join(runresultdir,
                                           'background_computationParameters.txt')

            # Count number of params
            paramfiles = [p for p in os.listdir(runresultdir) if 'parameter0' in p]

            # Human-sort list
            #https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
            paramfiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            print('Number of parameters', len(paramfiles))

            # Make A4 figure
            pdffigure = os.path.join(runresultdir,
                                     'evaluationplot_' + star + '_' + run + '.pdf')

            success, params, model_name = make_pdffigure(pdffigure=pdffigure,
                                                         datafile=datafile,
                                                         computationfile=computationfile,
                                                         summaryfile=summaryfile,
                                                         paramfiles=paramfiles,
                                                         resultdir=resultdir,
                                                         runresultdir=runresultdir)

            # Open pdf page with prompt
            #subprocess.call(["xdg-open", pdffigure])
            webbrowser.open_new_tab(pdffigure)

            print('What is your evaluation?')
            if success:
                print('If you want to keep it, write something in', goodinput)
            print('If you want to reset it with new parameters, write something in', retryinput)
            print('If you want to fully discard the star, write something in', discardinput)
            if success:
                usereval = sanitised_input(range_=fullinput)
            else:
                usereval = sanitised_input(range_=retryinput + discardinput)

            if success and usereval in goodinput:
                print('Evaluation: Good fit')
                print('Fitting directory is copied to', goodevaldir)
                copy_tree(runresultdir, os.path.join(goodevaldir, star))
            elif usereval in retryinput:
                print('Evalutation: Retry fit')
                newhyperparams = os.path.join(resultdir,
                                              'background_hyperParameters_' + newrun + '.txt')
                print('Would you like to keep the model? y/n')
                print('The current model is', model_name)
                usermodelchoice = sanitised_input(type_=str, range_=yninput)
                if usermodelchoice in noinput:
                    while True:
                        print('Which model would you like to change to?')
                        print('Choose one in', backgroundmodels)
                        usermodel = sanitised_input(type_=str)
                        if usermodel not in backgroundmodels:
                            print('Choose a model in', backgroundmodels)
                            continue
                        else:
                            model_name = usermodel
                            run_numaxmode(star, model_name, newrun)
                            break
                else:
                    print('Would you like to change the parameters manually (`m`) or based on a numax (`n`)?')
                    userretrymode = sanitised_input(range_=manualmode + numaxmode)
                    if userretrymode in numaxmode:
                        run_numaxmode(star, model_name, newrun)
                    else:
                        print('How would you like to update the parameters?')
                        hyperparamsfile = os.path.join(resultdir,
                                                       'background_hyperParameters_' + run + '.txt')
                        newhyperparamsfile = os.path.join(resultdir,
                                                          'background_hyperParameters_' + newrun + '.txt')
                        hyperparams = np.loadtxt(hyperparamsfile, skiprows=6, unpack=True).T
                        params = []
                        for i, line in enumerate(hyperparams):
                            print(f'For parameter {i}:')
                            print('The current range is:')
                            print(line)
                            print('Do you want to change it? y/n')
                            userchangeparam = sanitised_input(range_=yninput)
                            if userchangeparam in goodinput:
                                while True:
                                    print('What do you want to change it to?')
                                    print('New minimum:')
                                    usernewparammin = sanitised_input(type_=float)
                                    print('New maximum:')
                                    usernewparammax = sanitised_input(type_=float)
                                    if usernewparammin > usernewparammax:
                                        print('Minimum must be less than maximum!')
                                        continue
                                    else:
                                        newline = [float(usernewparammin),
                                                   float(usernewparammax)]
                                        params.extend(newline)
                                        break
                            else:
                                print('We keep it')
                                print(line)
                                params.extend([line[0].astype(float),
                                               line[1].astype(float)])
                        header = """
                        Hyper parameters used for setting up uniform priors.
                        Each line corresponds to a different free parameter (coordinate).
                        Column #1: Minima (lower boundaries)
                        Column #2: Maxima (upper boundaries)
                        """
                        params = np.asarray(params)
                        params = np.reshape(params, ((len(params) // 2) , 2))
                        np.savetxt(newhyperparamsfile, params, fmt='%.3f', header=header)
                        print('New hyperparameters can be found in', newhyperparamsfile)
            elif usereval in discardinput:
                print('Star is discarded')
                print('Star has been added to the discard list', discardfile)
                with open(discardfile, 'a') as dl:
                    dl.write(star + '\n')
            else:
                # This should never run when fullinput only has three options
                print('Input not recognised, try input in ',
                      goodinput + discardinput + retryinput)

            print('Do you want to add any notes? y/n')
            usernotes = sanitised_input()
            if usernotes in goodinput:
                print('Write your notes here:')
                usernotes = sanitised_input()
            else:
                usernotes = "None"

            writer.writerow({'date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                             'star': star,
                             'decision': usereval,
                             'notes': usernotes,
                             'model': model_name,
                             'params': params})
    make_retryshellscript(idstr=idstr, run=run)


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['eval', 'retry'])
try:
    if __name__ == "__main__":
        args = parser.parse_args()
        if args.mode == "eval":
            evaluate()
            print('\n')
            print('Thanks for now -- bye!')
        else:
            make_retryshellscript(idstr=None, run=None)
            print('\n')
            print('Thanks for now -- bye!')
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
