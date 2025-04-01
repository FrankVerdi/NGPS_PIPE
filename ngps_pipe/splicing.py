"""
Automated splicing for P200 NGPS.
"""

import os
from typing import Tuple, List

import numpy as np

from astropy.io import fits
import astropy.stats
import astropy.table

import pypeit
import pypeit.pypeit
from pypeit import msgs

import ngps_pipe

def get_raw_hdus_from_spec1d(spec1d_list: List[Tuple[str, int]], root: str,
        output_path: str) -> List[fits.BinTableHDU]:
    """
    Returns list of ``fits.BinTableHDU`` s, each containing the raw header and
    the 1D spectrum, of the input spec1d files.

    Args:
        spec1d_list (List[Tuple[str, int]]): List of (spec1d filename, spatial
            pixel coordinate)
        root (str): Path to raw data files, possibly including filename prefix.
        output_path (str): reduction output path

    Returns:
        List[fits.BinTableHDU]: List of raw data headers and data from input
            spec1d files.
    """
    ret = []
    for (spec1d, spat) in spec1d_list:
        raw_fname = os.path.join(root, f"{'_'.join(spec1d.split('_')[1:3])}_{spec1d.split('_')[3].split('-')[0]}.fits")
        print(raw_fname)

        # get the raw header
        with fits.open(raw_fname) as raw_hdul:
            raw_header = raw_hdul[0].header.copy()
        # get the spectrum
        spec1d_path = os.path.join(output_path, 'Science', spec1d)
        with fits.open(spec1d_path) as spec1d_hdul:
            for hdu in spec1d_hdul:
                if f'SPAT{spat:04d}' in hdu.name:
                    raw_data = hdu.data.copy()
        wave_col = fits.Column(name='wave', array=raw_data['OPT_WAVE'], unit='ANGSTROM', format='D')
        flux_col = fits.Column(name='flux', array=raw_data['OPT_FLAM'], unit='E-17 ERG/S/CM^2/ANG', format='D')
        sigma_col = fits.Column(name='sigma', array=raw_data['OPT_FLAM_SIG'], unit='E-17 ERG/S/CM^2/ANG', format='D')
        ret.append(fits.BinTableHDU.from_columns([wave_col, flux_col, sigma_col],
            name=os.path.splitext(os.path.basename(raw_fname))[0].upper(),
            header=raw_header))
    return ret


def splice(splicing_dict: dict, interpolate_gaps: bool, root: str, output_path_I: str, output_path_R: str, spliced_path: str) -> None:
    """
    Splices red and blue spectra together.

    .. code-block::

        splicing_dict[target_name][position_along_slit][arm] = {
            'spec1ds': [(spec1d_filename_1, spatial_pixel_1), (spec1d_filename_2, spatial_pixel_2)],
            'coadd': coadd_filename
        }

    Args:
        splicing_dict (dict): Guides splicing.
        interpolate_gaps (bool): Interpolate across gaps in wavelength coverage?
        root (str): Path to raw data files, possibly including filename prefix.
        output_path (str): reduction output path
    """
    for target, targets_dict in splicing_dict.items():
        label = 'a'
        for _, arm_dict in targets_dict.items():
            I_dict = arm_dict.get('I', {})
            R_dict = arm_dict.get('R', {})

            Rfile = R_dict.get('coadd')
            Ifile = I_dict.get('coadd')
            spec_R = None
            spec_I = None
            if Rfile is not None:
                Rfile = os.path.join(output_path_R, 'Science', Rfile)
                spec_R = fits.open(Rfile)[1].data
            if Ifile is not None:
                Ifile = os.path.join(output_path_I, 'Science', Ifile)
                spec_I = fits.open(Ifile)[1].data
            if Rfile is None and Ifile is None:
                continue

            ((final_wvs, final_flam, final_flam_sig),
                (I_wvs, I_flam, I_sig),
                (R_wvs, R_flam, R_sig)) = adjust_and_combine_overlap(spec_R, spec_I, interpolate_gaps)

            primary_header = fits.Header()
            primary_header['HIERARCH NGPS_PIPE_V'] = ngps_pipe.__version__
            primary_header['PYPEIT_V'] = pypeit.__version__
            primary_header['NUMPY_V'] = np.__version__
            primary_header['HIERARCH ASTROPY_V'] = astropy.__version__
            primary_header['B_COADD'] = Rfile
            primary_header['R_COADD'] = Ifile
            primary_hdu = fits.PrimaryHDU(header=primary_header)

            raw_I_hdus = get_raw_hdus_from_spec1d(I_dict.get('spec1ds', []), root, output_path_I)
            raw_R_hdus = get_raw_hdus_from_spec1d(R_dict.get('spec1ds', []), root, output_path_R)


            col_wvs = fits.Column(name='wave', array=I_wvs, unit='ANGSTROM', format='D')
            col_flux = fits.Column(name='flux', array=I_flam, unit='E-17 ERG/S/CM^2/ANG', format='D')
            col_error = fits.Column(name='sigma', array=I_sig, unit='E-17 ERG/S/CM^2/ANG', format='D')
            I_hdu = fits.BinTableHDU.from_columns([col_wvs, col_flux, col_error], name="I")

            col_wvs = fits.Column(name='wave', array=R_wvs, unit='ANGSTROM', format='D')
            col_flux = fits.Column(name='flux', array=R_flam, unit='E-17 ERG/S/CM^2/ANG', format='D')
            col_error = fits.Column(name='sigma', array=R_sig, unit='E-17 ERG/S/CM^2/ANG', format='D')
            R_hdu = fits.BinTableHDU.from_columns([col_wvs, col_flux, col_error], name="R")

            col_wvs = fits.Column(name='wave', array=final_wvs, unit='ANGSTROM', format='D')
            col_flux = fits.Column(name='flux', array=final_flam, unit='E-17 ERG/S/CM^2/ANG', format='D')
            col_error = fits.Column(name='sigma', array=final_flam_sig, unit='E-17 ERG/S/CM^2/ANG', format='D')
            table_hdu = fits.BinTableHDU.from_columns([col_wvs, col_flux, col_error], name="SPLICED")

            table_hdu.header['HIERARCH INTERP_GAPS'] = interpolate_gaps

            hdul = fits.HDUList(hdus=[primary_hdu, *raw_I_hdus, *raw_R_hdus, I_hdu, R_hdu, table_hdu])

            log_msg = f"{target}_{label}.fits contains "
            if Ifile is None:
                log_msg += f"{os.path.basename(Rfile)}"
            elif Rfile is None:
                log_msg += f"{os.path.basename(Ifile)}"
            else:
                log_msg += f"{os.path.basename(Ifile)} and {os.path.basename(Rfile)}"
            print(log_msg)

            hdul.writeto(os.path.join(spliced_path, f'{target}_{label}.fits'), overwrite=True)
            label = chr(ord(label) + 1)

def adjust_and_combine_overlap(
    spec_R: fits.FITS_rec,
    spec_I: fits.FITS_rec,
    interpolate_gaps: bool,
    I_mult: float = 1.0
) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Takes in red and blue spectra, adjusts overall flux level by red_mult, and
    combines overlap region.

    In the overlap region, the red spectrum is linearly interpolated to match
    the blue spectrum's wavelength spacing.

    Args:
        spec_b (fits.FITS_rec): blue spectrum
        spec_r (fits.FITS_rec): red spectrum.
        interpolate_gaps (bool): Interpolate across gaps in wavelength coverage?
        red_mult (float, optional): Factor multiplied into the red spectrum to
            match overal flux level with the blue spectrum. Defaults to 1.0.

    Raises:
        ValueError: Raised when both `spec_b` and `spec_r` are empty or None.

    Returns:
        Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ]: (blue, red, combined) spectra, where each spectrum is a tuple of
            (wavelengths, flux, error)
    """
    if ((spec_R is None or not spec_R['wave'].shape[0]) and
        (spec_I is None or not spec_I['wave'].shape[0])):
        raise ValueError("Both arguments cannot be empty or None.")
    # TODO: propagate input masks
    if spec_I is None or not spec_I['wave'].shape[0]:
        return ((spec_R['wave'], spec_R['flux'], spec_R['ivar'] ** -0.5),
                (None, None, None),
                (spec_R['wave'], spec_R['flux'], spec_R['ivar'] ** -0.5))
    if spec_R is None or not spec_R['wave'].shape[0]:
        return ((spec_I['wave'], I_mult*spec_I['flux'], I_mult*spec_I['ivar'] ** -0.5),
                (spec_I['wave'], spec_I['flux'], spec_I['ivar'] ** -0.5),
                (None, None, None))

    # combination steps
    overlap_lo = spec_I['wave'][0]
    overlap_hi = spec_R['wave'][-1]
    # maybe need to fix the overlaps?
    # need more finely spaced grid to be completely contained within coarser grid

    if overlap_lo > overlap_hi:
        # there is no overlap!
        # we can't adjust the flux level
        # so we just concatenate!
        final_wvs = np.concatenate([spec_R['wave'], spec_I['wave']])
        final_flam = np.concatenate([spec_R['flux'], spec_I['flux']*I_mult])
        final_flam_sig = np.concatenate([spec_R['ivar'] ** -0.5, (spec_I['ivar'] ** -0.5) * I_mult])
        return ((final_wvs, final_flam, final_flam_sig),
            (spec_I['wave'], spec_I['flux'], spec_I['ivar'] ** -0.5),
            (spec_R['wave'], spec_R['flux'], spec_R['ivar'] ** -0.5))

    olap_I = (spec_I['wave'] < overlap_hi)
    olap_R = (spec_R['wave'] > overlap_lo)

    ## 05/25/2021 red_mult is not really necessary, spectra look better without it.
    ## 06/25/2021 keeping red_mult as an argument for manual_splicing
    #red_mult = (astropy.stats.sigma_clipped_stats(spec_b['flux'][olap_b])[1] /
    #    astropy.stats.sigma_clipped_stats(spec_r['flux'][olap_r])[1])


    # different dispersion.
    wvs_R = spec_R['wave'][~olap_R]
    wvs_I = spec_I['wave'][~olap_I]
    flam_R = spec_R['flux'][~olap_R]
    flam_I = spec_I['flux'][~olap_I]
    flam_sig_R = spec_R['ivar'][~olap_R] ** -0.5
    flam_sig_I = spec_I['ivar'][~olap_I] ** -0.5


    olap_wvs_I = spec_I['wave'][olap_I]
    olap_flam_I = I_mult * spec_I['flux'][olap_I]
    olap_flam_sig_I = I_mult * spec_I['ivar'][olap_I] ** -0.5
    olap_wvs_R = spec_R['wave'][olap_R][:-1]
    olap_flam_R = spec_R['flux'][olap_R][:-1]
    olap_flam_sig_R = spec_R['ivar'][olap_R][:-1] ** -0.5

    olap_flam_I_interp, olap_flam_sig_I_interp = interp_w_error(olap_wvs_R, olap_wvs_I, olap_flam_I, olap_flam_sig_I, interpolate_gaps)

    olap_flams = np.array([olap_flam_I_interp, olap_flam_R])
    sigs = np.array([olap_flam_sig_I_interp, olap_flam_sig_R])
    weights = sigs ** -2.0

    olap_flam_avgd = np.average(olap_flams, axis=0, weights=weights)
    olap_flam_sig_avgd = 1.0 / np.sqrt(np.mean(weights, axis=0))

    final_wvs = np.concatenate((wvs_R, olap_wvs_R, wvs_I))
    final_flam = np.concatenate((flam_R, olap_flam_avgd, I_mult * flam_I))
    final_flam_sig = np.concatenate((flam_sig_R, olap_flam_sig_avgd, I_mult * flam_sig_I))

    return ((final_wvs, final_flam, final_flam_sig),
        (spec_I['wave'], spec_I['flux'], spec_I['ivar'] ** -0.5),
        (spec_R['wave'], spec_R['flux'], spec_R['ivar'] ** -0.5))

def interp_w_error(x: np.ndarray, xp: np.ndarray, yp: np.ndarray,
    err_yp: np.ndarray, interpolate_gaps: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate the data points (``xp``, ``yp``) with ``err_yp``
    uncertainty onto the grid ``x``.

    Args:
        x (np.ndarray): destination x data
        xp (np.ndarray): source x data
        yp (np.ndarray): source y data
        err_yp (np.ndarray): source y error data
        interpolate_gaps (bool): Interpolate across gaps in ``xp``?

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated y and error.
    """
    if len(xp) == 1:
        return np.ones_like(x) * yp[0], np.ones_like(x) * err_yp[0]

    y = np.zeros_like(x)
    yerr = np.zeros_like(x)
    slopes = np.zeros(xp.shape[0] - 1)

    dxp = np.diff(xp)
    mean_dxp, _, _ = astropy.stats.sigma_clipped_stats(dxp)

    for i in range(len(slopes)):
        slopes[i] = (yp[i+1] - yp[i])/dxp[i]
    #slopes[-1] = slopes[-2]

    for i in range(len(x)):
        # find the index j into xp such that xp[j-1] <= x[i] < xp[j]
        j = np.searchsorted(xp, x[i], side='right')
        if (x[i] == xp[j-1]):
            y[i] = yp[j-1]
            yerr[i] = err_yp[j-1]
        elif (j == len(xp)):
            # extrapolating outside domain!!!
            y[i] = yp[-1]# + slopes[j-2]*(x[i] - xp[-1])
            yerr[i] = np.sqrt((((x[i] - xp[-2])*err_yp[-1]) ** 2 + ((x[i] - xp[-1])*err_yp[-2]) ** 2) / ((xp[-2] - xp[-1]) ** 2))
        elif (j == 0):
            # extrapolating outside domain!!!
            y[i] = yp[0]# + slopes[j]*(x[i] - xp[0])
            yerr[i] = np.sqrt((((x[i] - xp[0])*err_yp[1]) ** 2 + ((x[i] - xp[1])*err_yp[0]) ** 2) / ((xp[1] - xp[0]) ** 2))
        else:
            y[i] = yp[j-1] + slopes[j-1]*(x[i] - xp[j-1])
            # If we are interpolating a gap larger than 5 times the avg d lambda
            if (xp[j] - xp[j-1]) > mean_dxp * 5:
                if interpolate_gaps:
                    # err is max of edge points
                    yerr[i] = max(err_yp[j], err_yp[j-1])
                else:
                    # err is infinite, so this point is completely discounted
                    yerr[i] = np.inf
            else:
                yerr[i] = np.sqrt((((x[i] - xp[j])*err_yp[j-1]) ** 2 + ((x[i] - xp[j-1])*err_yp[j]) ** 2) / ((xp[j-1] - xp[j]) ** 2))
    return y, yerr
