import os
import sys
import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table, join, vstack
from matplotlib import pyplot as plt
from astropy import units as u

sys.path.insert(0, "./results/python")
import background as bg


def deltanu_lightkurve(frequency, power, numax, floatfill=-99, silent=False):
    """Returns the average value of the large frequency spacing, DeltaNu,
    of the seismic oscillations of the target, using an autocorrelation
    function.

    There are many papers on the topic of autocorrelation functions for
    estimating seismic parameters, including but not limited to:
    Roxburgh & Vorontsov (2006), Roxburgh (2009), Mosser & Appourchaux (2009),
    Huber et al. (2009), Verner & Roxburgh (2011) & Viani et al. (2019).

    We base this approach first and foremost off the approach taken in
    Mosser & Appourchaux (2009). Given a known numax, a window around this
    numax is taken of one estimated full-width-half-maximum (FWHM) of the
    seismic mode envelope either side of numax. This width is chosen so that
    the autocorrelation includes all of the visible mode peaks.

    The autocorrelation (numpy.correlate) is given as:

    C = sum(s * s)

    where s is a window of the signal-to-noise spectrum. When shifting
    the spectrum over itself, C will increase when two mode peaks are
    overlapping. Because of the method of this calculation, we need to first
    rescale the power by subtracting its mean, placing its mean around 0. This
    decreases the noise levels in the ACF, as the autocorrelation of the noise
    with itself will be close to zero.

    As is done in Mosser & Appourchaux, we rescale the value of C in terms
    of the noise level in the ACF spectrum as

    A = (|C^2| / |C[0]^2|) * (2 * len(C) / 3) .

    The method will autocorrelate the region around the estimated numax
    expected to contain seismic oscillation modes. Repeating peaks in the
    autocorrelation implies an evenly spaced structure of modes.
    The peak closest to an empirical estimate of deltanu is taken as the true
    value. The peak finding algorithm is limited by a minimum spacing
    between peaks of 0.5 times the empirical value for deltanu.

    Our empirical estimate for numax is taken from Stello et al. (2009) as

    deltanu = 0.294 * numax^0.772

    If `numax` is None, a numax is calculated using the estimate_numax()
    function with default settings.

    NOTE: This function is intended for use with solar like Main Sequence
    and Red Giant Branch oscillators only.

    Parameters:
    ----------
    numax : float
        An estimated numax value of the mode envelope in the periodogram. If
        not given units it is assumed to be in units of the periodogram
        frequency attribute.

    Returns:
    -------
    deltanu : `SeismologyQuantity`
        The average large frequency spacing of the seismic oscillation modes.
        In units of the periodogram frequency attribute.
    """

    # Run some checks on the passed in numaxs
    # Ensure input numax is in the correct units
    frequency = u.Quantity(frequency, u.microhertz)
    ppm = u.def_unit(["ppm", 'parts per million'], u.Unit(1e-6))
    u.add_enabled_units(ppm)
    numax = u.Quantity(numax, frequency.unit)
    fs = np.median(np.diff(frequency.value))
    if numax.value  <  fs:
        raise ValueError("The input numax can not be lower than"
                        " a single frequency bin.")
    if numax.value > np.nanmax(frequency.value):
        raise ValueError("The input numax can not be higher than"
                        "the highest frequency value in the periodogram.")

    # Calculate deltanu using the method by Stello et al. 2009
    # Make sure that this relation only ever happens in microhertz space
    deltanu_emp = u.Quantity((0.294 * u.Quantity(numax, u.microhertz).value ** 0.772)
            * u.microhertz, frequency.unit).value

    # utils.get_fwhm
    if u.Quantity(frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
        fwhm = 0.25 * numax.value
    else:
        fwhm = 0.66 * numax.value**0.88
    # Changed from floor to ceil
    window_width = 2*int(np.ceil(fwhm))

    # utils.autocorrelate
    frequency_spacing = np.median(np.diff(frequency.value))

    spread = int(window_width/2/frequency_spacing)  # Find the spread in indices
    assert spread > 0, print(frequency_spacing, window_width)
    x = int(numax.value / frequency_spacing)  # Find the index value of numax
    x0 = int((frequency[0].value/frequency_spacing))    # Transform in case the index isn't from 0
    xt = x - x0
    # Avoid selecting index below 0
    ws_lower = np.maximum(xt-spread, 0)
    ws_upper = np.minimum(xt+spread, len(power))
    p_sel = copy.deepcopy(power[ws_lower:ws_upper])       # Make the window selection
    p_sel -= np.nanmean(p_sel)    #Make it so that the selection has zero mean.

    aacf = np.correlate(p_sel, p_sel, mode='full')[len(p_sel)-1:]     #Correlated the resulting SNR space with itself

    acf = (np.abs(aacf**2)/np.abs(aacf[0]**2)) / (3/(2*len(aacf)))
    fs = np.median(np.diff(frequency.value))
    lags = np.linspace(0., len(acf)*fs, len(acf))

    #Select a 50% region region around the empirical deltanu
    th = .5
    try:
        sel = (lags > deltanu_emp - th*deltanu_emp) & (lags  <  deltanu_emp + th*deltanu_emp)

        #Run a peak finder on this region
        peaks, _ = find_peaks(acf[sel], distance=np.floor(deltanu_emp/2. / fs))

        #Select the peak closest to the empirical value
        best_deltanu_value = lags[sel][peaks][np.argmin(np.abs(lags[sel][peaks] - deltanu_emp))]
    except ValueError:
        try:
            sel = (lags > deltanu_emp - 2*deltanu_emp) & (lags  <  deltanu_emp + 2*deltanu_emp)

            #Run a peak finder on this region
            peaks, _ = find_peaks(acf[sel], distance=np.floor(deltanu_emp/2. / fs))

            #Select the peak closest to the empirical value
            best_deltanu_value = lags[sel][peaks][np.argmin(np.abs(lags[sel][peaks] - deltanu_emp))]
        except ValueError:
            return floatfill

    best_deltanu = u.Quantity(best_deltanu_value, frequency.unit)
    diagnostics = {'lags':lags, 'acf':acf, 'peaks':peaks, 'sel':sel, 'numax':numax, 'deltanu_emp':deltanu_emp}
    if not silent:
        print('Using Lightkurve estimator, deltaNu is ', best_deltanu)
    return best_deltanu.value


def deltanu_famed(frequency, power, numax, numax_le, numax_ue, sigma,
        floatfill=-99, runtimeerror=-88, silent=False, debugplot=False):
    # ;----------------------------------------------------------------------
    # Based on the following IDL routine
    # ; AUTHOR: Enrico Corsaro
    # ; INSTITUTION: INAF OACT
    # ;
    # ; PURPOSE: Computes an estimate of DeltaNu from an autocorrelation
    # ;          function of the stellar power spectral density centered
    # ;          around an input value of nuMax.
    # ;----------------------------------------------------------------------

    # crush_coeff from extract_global_parameters
    tmp_crush = (((numax - (3 * sigma)) < frequency) &
            (frequency < (numax + (3 * sigma))))
    crush_coeff = (
            ((np.sum((frequency[tmp_crush] - numax) ** 4)
                / np.sum(tmp_crush))
            / (sigma ** 4)))
    if crush_coeff < 10:
        tmp = ((frequency > (numax - (2 * sigma))) & (frequency < (numax + (2 * sigma))))
    else:
        tmp = ((frequency > (numax - (3 * sigma))) & (frequency < (numax + (3 * sigma))))

    lower_bound = numax - sigma
    upper_bound = numax + sigma

    # dnu_calc
    if numax > 300:
        alpha = 0.22
        bet = 0.797
    else:
        alpha = 0.267
        bet = 0.760
    guess_dnu = alpha * (numax ** bet)

    lower_bound = np.minimum(lower_bound, numax - (guess_dnu * 2))
    upper_bound = np.maximum(upper_bound, numax + (guess_dnu * 2))

    tmp = ((frequency > lower_bound) & (frequency < upper_bound))

    psd_cut = power[tmp]
    freq_cut = frequency[tmp]

    top_dnu = guess_dnu * 1.6
    bottom_dnu = guess_dnu / 1.5

    if not silent:
        print('Expected DeltaNu from scaling rel.: ', guess_dnu)
        print(f'Inspecting DeltaNu range: {bottom_dnu}--{top_dnu} microhertz')

    freqbin = frequency[1] - frequency[0]
    dnu_range_bins = round((top_dnu - bottom_dnu) / freqbin)
    bottom_dnu_bins = round(bottom_dnu / freqbin)
    lag = np.arange(dnu_range_bins) + bottom_dnu_bins

    if lag[-1] > len(psd_cut):
        lag = lag[lag < len(psd_cut)]

    width = guess_dnu / freqbin
    win_len= int(width)
    if win_len % 2 == 0:
        win_len += 1
    psd_smth = bg.smooth(power, window_len=win_len, window='flat')
    psd_cut_smth = psd_smth[tmp]

    result1 = a_correlate(psd_cut, lag)
    result2 = a_correlate(psd_cut_smth, lag)
    result1 -= np.amin(result1)
    result2 -= np.amin(result2)

    best_acf1 = np.amax(result1)
    best_dnu1 = freqbin * lag[np.argmax(result1)]
    best_acf2 = np.amax(result2)
    best_dnu2 = freqbin * lag[np.argmax(result2)]

    numax_pred1 = numax_calc_famed(best_dnu1)
    numax_pred2 = numax_calc_famed(best_dnu2)

    if not silent:
        print("Best DeltaNu (max ACF of PSD): ", best_dnu1)
        print("Expected nuMax (predicted 1): ", numax_pred1)
        print("Best DeltaNu (max ACF of smoothed PSD): ", best_dnu2)
        print("Expected nuMax (predicted 2): ", numax_pred2)

    # Perform a Gaussian fit to the ACF^2 from the smoothed PSD
    dnu_acf = lag * freqbin
    x = dnu_acf
    y = result2 ** 2

    if len(x) < 5:
        return floatfill, floatfill, best_dnu2

    if len(x) == 5:
        try:
            coeff, _ = curve_fit(
                    gauss_nterm4,
                    x, y,
                    p0=[np.amax(y), best_dnu2, 0.05 * best_dnu2, np.amin(y)])
        except RuntimeError:
            return runtimeerror, runtimeerror, best_dnu2
    elif len(x) == 6:
        try:
            coeff, _ = curve_fit(
                    gauss_nterm5,
                    x, y,
                    p0=[np.amax(y), best_dnu2, 0.05 * best_dnu2, np.amin(y), 0])
        except RuntimeError:
            return runtimeerror, runtimeerror, best_dnu2
    else:
        try:
            coeff, _ = curve_fit(
                    gauss_nterm6,
                    x, y,
                    p0=[np.amax(y), best_dnu2, 0.05 * best_dnu2, np.amin(y), 0, 0])
        except RuntimeError:
            return runtimeerror, runtimeerror, best_dnu2

    x2 = x
    y2 = y
    estimates = [coeff[0], coeff[1], abs(coeff[2])]

    diff1 = np.abs(coeff[1] - best_dnu2)

    if diff1 > (np.amax(dnu_acf) - np.amin(dnu_acf))/2:
        tmp_range = ((x < (best_dnu2 * 1.1)) & (x > (best_dnu2 * 0.9)))

        if np.sum(tmp_range) > 3:
            x2 = x[tmp_range]
            y2 = y[tmp_range]
            estimates = [np.amin(y2), best_dnu2, 0.05 * best_dnu2]

    try:
        coeff2, _ = curve_fit(
                gauss_nterm3,
                x2, y2,
                p0=estimates)
    except RuntimeError:
        return runtimeerror, runtimeerror, best_dnu2

    if debugplot:
        plt.figure()
        plt.xlabel('Frequency close to numax')
        plt.ylabel('Power')
        plt.scatter(x2, y2, s=3, color='k')
        plt.plot(x2, gauss_nterm3(x2, *coeff2))
        plt.axvline(coeff[1], color='g', label='First iteration Gauss')
        plt.axvline(coeff2[1], color='b', label='Second iteration Gauss')
        plt.axvline(guess_dnu, color='r', label='Guess')
        plt.legend()
        plt.savefig('debug.png')
    diff2 = np.abs(coeff2[1] - best_dnu2)

    if (diff1 < diff2) and (coeff[2] > 0):
        dnu_acf = coeff[1]
        dnu_acf_sig = coeff[2]
    else:
        dnu_acf = coeff2[1]
        dnu_acf_sig = coeff2[2]

    if not silent:
        print('Best DeltaNu from Gaussian fit to ACF^2:', dnu_acf)
        print('DeltaNu (sig):', dnu_acf_sig)
    return dnu_acf, dnu_acf_sig, best_dnu2


def gauss_nterm6(x, a0, a1, a2, a3, a4, a5):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2) + a3 + a4 * x + a5 * x**2
    return y

def gauss_nterm5(x, a0, a1, a2, a3, a4):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2) + a3 + a4 * x
    return y

def gauss_nterm4(x, a0, a1, a2, a3):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2) + a3
    return y


def gauss_nterm3(x, a0, a1, a2):
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2)
    return y


def a_correlate(y, lag):
    #https://stackoverflow.com/a/73026983/1570972
    y = np.asarray(y)
    lag = np.asarray(lag)
    n = len(y)
    yunbiased = y - np.mean(y)
    ynorm = np.sum(yunbiased**2)
    r = np.correlate(yunbiased, yunbiased, "full") / ynorm
    return r[lag + (n - 1)]


def numax_calc_famed(dnu):
    if dnu > 20:
        alpha = 0.22
        bet = 0.797
    else:
        alpha = 0.267
        bet = 0.760
    return (dnu / alpha) ** (1/bet)


def extract_parameters(star_id, bgrun, returnpsd=False):
    if bgrun in ['05', '06']:
        datadir = './data'
    elif bgrun in ['03', '04']:
        datadir = './data_multi'
    elif bgrun in ['00', '01', '02']:
        datadir = './data_allfields'
    elif bgrun in ['0--', '']:
        if not returnpsd:
            return []
        datadir = './data_allfields'
        psd = os.path.join(datadir, star_id + '.txt')
        if not os.path.isfile(psd):
            datadir = './data'
    else:
        print(bgrun)
    psd = os.path.join(datadir, star_id + '.txt')
    results_dir = './results/' + star_id + '/' + bgrun + '/'
    freq, psd = np.loadtxt(psd, unpack=True)
    if returnpsd:
        return freq, psd
    # read numax, numax_le, numax_ue, sigma
    sumf = os.path.join(results_dir, 'background_parameterSummary.txt')
    medianpar,lowerpar,upperpar = np.loadtxt(sumf,
            unpack=True,
            usecols=(1,4,5))
    n_par = len(medianpar)
    numax = medianpar[n_par-2]
    numax_le = lowerpar[n_par-2]
    numax_ue = upperpar[n_par-2]
    sigma = medianpar[n_par-1]
    return [freq, psd, numax, numax_le, numax_ue, sigma]


if __name__ =='__main__':
    star_id = 'CRT0102565990'
    bgrun = '05'
    e = extract_parameters(star_id, bgrun)
    dnu_famed, error_dnu_famed = deltanu_famed(*e)
    dnu_lk = deltanu_lightkurve(*e[0:3])
