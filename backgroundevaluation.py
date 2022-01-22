# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate background fits for all stars in ./data/
# amalie
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import csv
import json
import os
import re
import shutil
import sys
import webbrowser
from datetime import date, datetime

import numpy as np
import pandas as pd

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
fieldnames = ['date', 'star', 'model', 'decision', 'notes', 'params']

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
    print('What is your estimate of numax?')
    usernumax = sanitised_input(type_=float)

    prefix = re.split('(\d+)', star)[0]
    star_id = re.split('(\d+)', star)[1]

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

    if os.path.isfile(pdffigure):
        return success, params, model_name

    with PdfPages(pdffigure) as pdf:
        fig = plt.figure(figsize=(8.27, 11.7)) # A4 format-ish
        ncols = 3
        gs = fig.add_gridspec(6, ncols)
        ax1 = fig.add_subplot(gs[0:2, :])

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


def make_retryshellscript(idstr, run, newrun=None, chunksize=300):
    # Read from logfile in order to properly handle resumed evaluations
    if newrun is None:
        newrun = str(int(run)+1)
        if len(newrun) < 2:
            newrun = '0' + newrun


    today = date.today().strftime('%Y%m%d')
    evaldir = './evaluation/'
    logfile = os.path.join(evaldir, idstr + 'run_' + run + '.csv')
    errorfile = 'errors_' + idstr + '_' + newrun + '.txt'
    if not os.path.isfile(logfile):
        print('Log file does not exist, exiting')
        return

    log = pd.read_csv(logfile, delimiter='\t')
    retrymask = ([d in retryinput for d in log['decision']])

    stars = log[retrymask]['star']
    models = log[retrymask]['model']
    if np.sum(retrymask) > 0:
        for i in range(0, len(log[retrymask]['star']), chunksize):
            retryshellscript = 'runBackground' + newrun + '_' + str(i) + '.sh'
            with open(retryshellscript, "w") as f:
                print("#!/bin/bash", file=f)
                print("""
                f() {
                        if ! ./background "$@" ; then
                                echo "[`date`] $@" >> %s
                        fi
                }""" % errorfile, file=f)
                print('', file=f)
                for star, model in zip(stars[i:i+chunksize], models[i:i+chunksize]):
                    prefix = re.split('(\d+)', star)[0]
                    star_id = re.split('(\d+)', star)[1]
                    assert model in backgroundmodels
                    print(f"f {prefix} {star_id} {newrun} {model} background_hyperParameters 0.0 0",
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


def get_idstr_run(idstr=None, run=None):
    if idstr is None:
        print('Please define an id-string')
        idstr = sanitised_input(type_=str)

    if run is None:
        print('Please define the run-string (e.g. 00 or 01)')
        run = sanitised_input(type_=int)
        run = str(run)
    if len(run) < 2:
        run = '0' + run
    return idstr, run


def do_eval(idstr, run):
    """
    This functions takes the actions made by evaluate() and performs them.
    This allows for undo's and take back's in evaluate().
    """
    today = date.today().strftime('%Y%m%d')
    datadir = './data/'
    resultdir = './results/'

    evaldir = './evaluation/'
    runevaldir = os.path.join(evaldir, idstr + '_run' + run)
    goodevaldir = os.path.join(runevaldir, 'goodfits')
    logfile = os.path.join(evaldir, idstr + 'run_' + run + '.csv')
    errorfile = os.path.join(evaldir, 'errors_' + run + '.txt')

    newrun = str(int(run)+1)
    if len(newrun) < 2:
        newrun = '0' + newrun

    df = pd.read_csv(logfile, delimiter='\t')

    goodmask = ([d in goodinput for d in df['decision']])
    retrymask = ([d in retryinput for d in df['decision']])

    print('For the %s stars with good evaluations, their fits will be copied to %s' % (
        np.sum(goodmask), goodevaldir))
    for star in df.star[goodmask]:
        resultdir = os.path.join('./results/' + star)
        runresultdir = os.path.join(resultdir, run)
        newrunresultdir = os.path.join(resultdir, run)
        copy_tree(runresultdir, os.path.join(goodevaldir, star))

    for star, params_str in zip(df.star[retrymask], df.params[retrymask]):
        resultdir = os.path.join('./results/' + star)
        params = json.loads(params_str)
        new_model_name = params["model"]
        if "numax" in params:
            run_numaxmode(star, new_model_name, newrun)
            prefix = re.split('(\d+)', star)[0]
            star_id = re.split('(\d+)', star)[1]
            bg.set_background_priors(
                catalog_id=prefix,
                star_id=star_id,
                numax=usernumax,
                model_name=model_name,
                dir_flag=int(newrun))
            print('set_background_priors has made a new hyperparameter file of run',
                  newrun)
        else:
            new_params = np.asarray(params["params"])

            header = """
            Hyper parameters used for setting up uniform priors.
            Each line corresponds to a different free parameter (coordinate).
            Column #1: Minima (lower boundaries)
            Column #2: Maxima (upper boundaries)
            """
            newrundir = os.path.join(resultdir, str(newrun))
            os.mkdir(newrundir)
            newhyperparamsfile = os.path.join(resultdir,
                                              'background_hyperParameters_' + newrun + '.txt')
            np.savetxt(newhyperparamsfile, new_params, fmt='%.3f', header=header)
            print('New hyperparameters can be found in', newhyperparamsfile)


def evaluate(idstr, run):
    today = date.today().strftime('%Y%m%d')
    datadir = './data/'
    resultdir = './results/'

    evaldir = './evaluation/'
    runevaldir = os.path.join(evaldir, idstr + '_run' + run)
    goodevaldir = os.path.join(runevaldir, 'goodfits')
    logfile = os.path.join(evaldir, idstr + 'run_' + run + '.csv')
    errorfile = os.path.join(evaldir, 'errors_' + run + '.txt')

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
            else:
                print('Evaluation is resumed')
                resume = True

    # Find all stars in datadir
    starlist = []
    for star in os.listdir(datadir):
        if star.endswith('.txt'):
            star = star.split('.')[0]
            datafile = os.path.join('./data/' + star + '.txt')
            resultdir = os.path.join('./results/' + star)
            runresultdir = os.path.join(resultdir, run)
            computationfile = os.path.join(runresultdir,
                                           'background_computationParameters.txt')
            if not os.path.exists(computationfile):
                continue
            starlist.append(star)

    assert len(starlist) > 0, 'No stars found - check ./data/'

    if resume:
        mode = 'a'
        df = pd.read_csv(logfile, delimiter='\t')
        for star in df.star:
            assert star in starlist, star
            print('Star is already evaluated:', star)
            starlist.remove(star)
    else:
        mode = 'w'
        print('Iterating over all stars in', datadir)

    # Make log
    with open(logfile, mode, newline='') as log:
        writer = csv.DictWriter(log, fieldnames=fieldnames, delimiter='\t')
        if mode == 'w':
            writer.writeheader()

        for i, star in enumerate(starlist):
            print('')
            print('Evaluating', star)
            print('%s out of %s' % (i, len(starlist)))
            # Navigate to directory
            datafile = os.path.join('./data/' + star + '.txt')
            resultdir = os.path.join('./results/' + star)
            runresultdir = os.path.join(resultdir, run)
            newrunresultdir = os.path.join(resultdir, run)

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

            log_row = {
                'date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                'star': star,
                'model': model_name,
                'decision': usereval,
            }

            if success and usereval in goodinput:
                print('Evaluation: Good fit')
                log_row['params'] = "None"
            elif usereval in retryinput:
                print('Evalutation: Retry fit')
                print('The current model is', model_name)
                print('Would you like to keep the model? y/n')
                usermodelchoice = sanitised_input(type_=str, range_=yninput)
                new_model_name = None

                if usermodelchoice in noinput:
                    while True:
                        print('Which model would you like to change to?')
                        print('Choose one in', backgroundmodels)
                        usermodel = sanitised_input(type_=str)
                        if usermodel not in backgroundmodels+['1', '2']:
                            print('Choose a model in', backgroundmodels)
                            continue
                        elif usermodel == '1':
                            new_model_name = 'OneHarvey'
                        elif usermodel == '2':
                            new_model_name = 'TwoHarvey'
                        else:
                            new_model_name = usermodel
                        break
                else:
                    print('Would you like to change the parameters manually (`m`) or based on a numax (`n`)?')
                    userretrymode = sanitised_input(range_=manualmode + numaxmode)
                    if userretrymode in numaxmode:
                        new_model_name = model_name

                if new_model_name is not None:
                    # Either we keep the model and run based on numax,
                    # or we change to another model.
                    print('What is your estimate of numax?')
                    usernumax = sanitised_input(type_=float)
                    log_row['params'] = json.dumps({"model": new_model_name, "numax": usernumax})
                else:
                    # Change the parameters manually

                    print('How would you like to update the parameters?')
                    hyperparamsfile = os.path.join(resultdir,
                                                   'background_hyperParameters_' + run + '.txt')
                    hyperparams = np.loadtxt(hyperparamsfile, skiprows=6, unpack=True).T
                    new_params = []
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
                                    new_params.extend(newline)
                                    break
                        else:
                            print('We keep it')
                            print(line)
                            new_params.extend([line[0].astype(float),
                                           line[1].astype(float)])
                    new_params = np.asarray(new_params)
                    new_params = np.reshape(new_params, ((len(new_params) // 2) , 2))
                    log_row["params"] = json.dumps({"model": model_name, "params": new_params.tolist()})
            elif usereval in discardinput:
                log_row['decision'] = "discard"
                log_row['params'] = "None"
                print('Star is discarded')
            else:
                # This should never run when fullinput only has three options
                raise Exception('Input not recognised, try input in %s' % (goodinput + discardinput + retryinput,))

            print('Do you want to add any notes? y/n')
            usernotes = sanitised_input(range_=yninput)
            if usernotes in goodinput:
                print('Write your notes here:')
                log_row["notes"] = sanitised_input()
            else:
                log_row["notes"] = "None"

            writer.writerow(log_row)

    print("Want to perform the actions from the evaluation (copy good + prepare reruns)?")
    print("If you say 'n' here, you can always run this script later with 'do' and 'retry'.")
    if sanitised_input(range_=yninput) in goodinput:
        do_eval(idstr, run)
        make_retryshellscript(idstr, run)


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['eval', 'do', 'retry'])
parser.add_argument("id", nargs="?")
parser.add_argument("run", nargs="?")


if __name__ == "__main__":
    args = parser.parse_args()
    print('Welcome to Background Evaluation')
    idstr, run = get_idstr_run(args.id, args.run)
    if args.mode == "eval":
        evaluate(idstr, run)
    elif args.mode == "do":
        do_eval(idstr, run)
    elif args.mode == "retry":
        make_retryshellscript(idstr, run)
    else:
        raise Exception("Unknown mode")
    print('\n')
    print('Thanks for now -- bye!')
