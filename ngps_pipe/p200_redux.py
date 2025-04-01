"""
Automatic Reduction Pipeline for P200 NGPS.
"""

import argparse
import os
import time
import multiprocessing
from typing import Optional, List
import pickle

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column, Row

import sys
sys.path.insert(0, "/Users/Frank/Desktop/NGPS/PypeIt") 
import pypeit

from pypeit.pypeitsetup import PypeItSetup
import pypeit.display
from pypeit.spectrographs.util import load_spectrograph

import tqdm

from ngps_pipe import reduction, qa, fluxing, coadding, telluric, splicing
from ngps_pipe import table_edit
from ngps_pipe import fix_headers


def entrypoint():
    main(parser())

def parser(options: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command line arguments

    Args:
        options (Optional[List[str]], optional): List of command line arguments.
            Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Define command line arguments.
    argparser = argparse.ArgumentParser(description="Automatic Data Reduction Pipeline for P200 NGPS",
        formatter_class=argparse.RawTextHelpFormatter)

    # Argument for fully-automatic (i.e. nightly) or with user-checking file typing
    argparser.add_argument('-i', '--no-interactive', default=False, action='store_true',
                           help='Interactive file-checking?')

    # Argument for input file directory
    argparser.add_argument('-r', '--root', type=os.path.abspath, default=None,
                           required=True,
                           help='File path+root, e.g. /data/NGPS_20200127')

    argparser.add_argument('-d', '--output_path', type=os.path.abspath,
                           default='.',
                           help='Path to top-level output directory.  '
                                'Default is the current working directory.')

    # Argument for specifying only red/blue

    argparser.add_argument('-a', '--arm', default=None, choices=['I','R'],
                           help='[I, R] to only reduce one arm (null splicing)')

    argparser.add_argument('-m', '--manual-extraction', default=False, action='store_true',
                           help='manual extraction')

    argparser.add_argument('--debug', default=False, action='store_true',
                           help='debug')
    argparser.add_argument('-j', '--jobs', type=int, default=1, metavar='N',
                            help='Number of processes to use')

    argparser.add_argument('-p', '--parameter-file', type=str, default="",
                           help="Path to parameter file. The parameter file should be formatted as follows:\n\n"
                            "[R]\n"
                            "** PypeIt parameters for the R side goes here **\n"
                            "[I]\n"
                            "** PypeIt parameters for the I side goes here **\n"
                            "EOF\n\n"
                            "The [I/R] parameter blocks are optional, and their order does not matter.")

    argparser.add_argument('-t', '--skip-telluric', default=False, action='store_true',
                           help='Skip telluric correction')

    argparser.add_argument('-c', '--null-coadd', default=False, action='store_true',
                           help="Don't coadd consecutive exposures of the same target.\n"
                                "By default consective exposures will be coadded.")

    argparser.add_argument('--splicing-interpolate-gaps', default=False, action='store_true',
                           help="Use this option to linearly interpolate across large gaps\n"
                                "in the spectrum during splicing. The default behavior is to\n"
                                "only use data from one detector in these gaps, which results\n"
                                "in a slightly noisier spliced spectrum.")

    return argparser.parse_args() if options is None else argparser.parse_args(options)

def interactive_correction(ps: PypeItSetup) -> None:
    """
    Allows for human correction of FITS headers and frame typing.

    Launches a GUI via ngps_pipe.table_edit, which handles saving updated FITS headers.
    table_edit depends on the current NGPS headers.

    Todo:
        Make table to FITS header mapping mutable

    Args:
        ps (PypeItSetup): PypeItSetup object created in ngps_pipe.reduction.setup
    """
    # function for interactively correcting the fits table
    fitstbl = ps.fitstbl
    deleted_files = []
    table_edit.main(fitstbl.table, deleted_files)
    files_to_remove = []
    for rm_file in deleted_files:
        for data_file in ps.file_list:
            if rm_file in data_file:
                files_to_remove.append(data_file)
                break
    for rm_file in files_to_remove:
        ps.file_list.remove(rm_file)

def main(args):

    t = time.perf_counter()


    output_path_R = os.path.join(args.output_path, 'redux_R')
    output_path_I = os.path.join(args.output_path, 'redux_I')

    if os.path.isdir(args.output_path):
        os.chdir(args.output_path)
    else:
        os.makedirs(args.output_path, exist_ok=True)

    if os.path.isdir(output_path_I):
        os.chdir(output_path_I)
    else:
        os.makedirs(output_path_I, exist_ok=True)

    if os.path.isdir(output_path_R):
        os.chdir(output_path_R)
    else:
        os.makedirs(output_path_R, exist_ok=True)

    if args.arm:
        do_R = args.arm.lower() == 'R'
        do_I = args.arm.lower() == 'I'
    else:
        do_R = True
        do_I = True


    qa_dict = {}
    qa_dict_R = {}
    qa_dict_I = {}

    if args.parameter_file:
        R_user_config_lines = reduction.parse_pypeit_parameter_file(args.parameter_file, 'p200_ngps_R')
        I_user_config_lines = reduction.parse_pypeit_parameter_file(args.parameter_file, 'p200_ngps_I')
    else:
        R_user_config_lines = []
        I_user_config_lines = []

    if args.debug:
        pypeit.display.display.connect_to_ginga(raise_err=True, allow_new=True)
    
    if do_I:
        I_files = fix_headers.main(args.root, args.no_interactive, args.no_interactive)
        context = reduction.setup(I_files, output_path_I, 'p200_ngps_I')   
        # optionally use interactive correction
        if not args.no_interactive:
            interactive_correction(context[0])
        pypeit_file_I = reduction.write_setup(context, 'all', 'p200_ngps_I', I_user_config_lines)[0]

    if do_R:
        R_files = fix_headers.main(args.root, args.no_interactive, args.no_interactive)
        context = reduction.setup(R_files, output_path_R, 'p200_ngps_R')
        if not args.no_interactive:
            interactive_correction(context[0])
        pypeit_file_R = reduction.write_setup(context, 'all', 'p200_ngps_R', R_user_config_lines)[0]

    ''''''
    plt.switch_backend("agg")
    # TODO: parallelize this
    # Would need to look like
    # Splitting up the .pypeit files into bits and pieces
    # Oooh what if I just do the calibration first
    # and then parallelize the reduction
    output_spec1ds_R = set()
    output_spec1ds_I = set()
    if do_I:
        output_spec1ds_I, output_spec2ds_I = reduction.redux(pypeit_file_I, output_path_I)
        #qa_dict_I = qa.save_2dspecs(qa_dict_I, output_spec2ds_I, output_path_I, 'p200_ngps_I')
        #qa.write_extraction_QA(qa_dict_I, output_path_I)
    if do_R:
        output_spec1ds_R, output_spec2ds_R = reduction.redux(pypeit_file_R, output_path_R)
        #qa_dict_R = qa.save_2dspecs(qa_dict_R, output_spec2ds_R, output_path_R, 'p200_ngps_R')
        #qa.write_extraction_QA(qa_dict_R, output_path_R)

    if do_I:
        verification_counter = 0
        I_pypeit_files = reduction.verify_spec1ds(output_spec1ds_I, verification_counter, output_path_I)
        while I_pypeit_files:
            verification_counter += 1

            out_1d, out_2d = reduction.re_redux(I_pypeit_files, output_path_I)
            I_pypeit_files = reduction.verify_spec1ds(out_1d, verification_counter, output_path_I)
            #qa_dict_I = qa.save_2dspecs(qa_dict_I, out_2d, output_path_I, 'p200_ngps_I')

            output_spec1ds_I |= out_1d
            output_spec2ds_I |= out_2d
    if do_R:
        verification_counter = 0
        R_pypeit_files = reduction.verify_spec1ds(output_spec1ds_R, verification_counter, output_path_R)
        while R_pypeit_files:
            verification_counter += 1

            out_1d, out_2d = reduction.re_redux(R_pypeit_files, output_path_R)
            R_pypeit_files = reduction.verify_spec1ds(out_1d, verification_counter, output_path_R)
            #qa_dict_R = qa.save_2dspecs(qa_dict_R, out_2d, output_path_R, 'p200_ngps_R')

            output_spec1ds_R |= out_1d
            output_spec2ds_R |= out_2d

    # TODO: use a do/while loop to iterate on the manual extraction GUI until user is satisfied
    if args.manual_extraction:
        # wait for user acknowledgement
        input("Ready for manual extraction? Make sure to check that $DISPLAY is correct and hit ENTER to continue")
        plt.switch_backend("Qt5Agg")

        if do_I:
            I_manual_pypeit_files = reduction.manual_extraction(output_spec2ds_I, pypeit_file_I, output_path_I)
        if do_R:
            R_manual_pypeit_files = reduction.manual_extraction(output_spec2ds_R, pypeit_file_R, output_path_R)
        if do_I and I_manual_pypeit_files:
            out_1d, out_2d = reduction.re_redux(I_manual_pypeit_files, output_path_I)
            #qa.save_2dspecs(qa_dict_I, out_2d, output_path_I, 'p200_ngps_I')

            output_spec1ds_I |= out_1d
            output_spec2ds_I |= out_2d
        if do_R and R_manual_pypeit_files:
            out_1d, out_2d = reduction.re_redux(R_manual_pypeit_files, output_path_R)
            #qa.save_2dspecs(qa_dict_R, out_2d, output_path_R, 'p200_ngps_R')

            output_spec1ds_R |= out_1d
            output_spec2ds_R |= out_2d

    # Find standards and make sensitivity functions
    spec1d_table = Table(names=('filename', 'arm', 'object', 'frametype',
                            'airmass', 'mjd', 'sensfunc', 'exptime'),
                         dtype=(f'U255', 'U4', 'U255', 'U8',
                            float, float, f'U255', float))

    spec1ds = output_spec1ds_I | output_spec1ds_R  
    for spec1d in spec1ds:
        if spec1d in output_spec1ds_I:
            arm = 'I'
            path = os.path.join(output_path_I, 'Science', spec1d) 
        else:
            arm = 'R'
            path = os.path.join(output_path_R, 'Science', spec1d) 

        with fits.open(path) as hdul:
            head0 = hdul[0].header
            head1 = hdul[1].header
            spec1d_table.add_row((spec1d, arm, head0['TARGET'],
                head1['OBJTYPE'], head0['AIRMASS'],
                head0['MJD'], '', head0['EXPTIME']))
            
    spec1d_table.add_index('filename')
    spec1d_table.sort(['arm', 'mjd'])

    if do_I:
        for row in spec1d_table[(spec1d_table['arm'] == 'I') & (spec1d_table['frametype'] == 'standard')]:
            sensfunc = fluxing.make_sensfunc(row['filename'], output_path_I, 'p200_ngps_I', I_user_config_lines)
            if sensfunc == "":
                spec1d_table['frametype'][spec1d_table['filename'] == row['filename']] = 'science'
            else:
                spec1d_table['sensfunc'][spec1d_table['filename'] == row['filename']] = sensfunc
    if do_R:
        for row in spec1d_table[(spec1d_table['arm'] == 'R') & (spec1d_table['frametype'] == 'standard')]:
            sensfunc = fluxing.make_sensfunc(row['filename'], output_path_R, 'p200_ngps_R', R_user_config_lines)
            if sensfunc == "":
                spec1d_table['frametype'][spec1d_table['filename'] == row['filename']] = 'science'
            else:
                spec1d_table['sensfunc'][spec1d_table['filename'] == row['filename']] = sensfunc

    if do_I:
        arm = spec1d_table['arm'] == 'I'

        stds = (spec1d_table['frametype'] == 'standard') & arm

        I_arm = load_spectrograph('p200_ngps_I')
        rawfile = os.path.join(args.root,
            '_'.join(spec1d_table[arm][0]['filename'].split('_')[1:4]).split('-')[0] + '.fits'
        )

        #config = '_'.join([
            #'I',
            #I_arm.get_meta_value(rawfile, 'dispname').replace('/', '_'),
            #I_arm.get_meta_value(rawfile, 'dichroic').lower()
        #])

        if np.any(stds):
            for row in spec1d_table[arm]:
                if row['frametype'] == 'science':
                    best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argmin()]
                elif row['frametype'] == 'standard':
                    if (stds).sum() == 1:
                        best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argmin()]
                    else:
                        best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argsort()[1]]
                spec1d_table.loc[row['filename']]['sensfunc'] = best_sens
        else:
            print("Could not find valid standard for sensitivity function, and no archived files found")
            #for filename in spec1d_table[arm]['filename']:
                #spec1d_table.loc[filename]['sensfunc'] = config
    if do_R:
        arm = spec1d_table['arm'] == 'R'
        stds = (spec1d_table['frametype'] == 'standard') & arm

        R_arm = load_spectrograph('p200_ngps_R')
        rawfile = os.path.join(args.root,
            '_'.join(spec1d_table[arm][0]['filename'].split('_')[1:4]).split('-')[0] + '.fits'
        )
        
        #config = '_'.join([
            #'R',
            #R_arm.get_meta_value(rawfile, 'dispname').replace('/', '_'),
            #R_arm.get_meta_value(rawfile, 'dichroic').lower()
        #])
        if np.any(stds):
            for row in spec1d_table[arm]:
                if row['frametype'] == 'science':
                    best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argmin()]
                elif row['frametype'] == 'standard':
                    if (stds).sum() == 1:
                        best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argmin()]
                    else:
                        best_sens = spec1d_table[stds]['sensfunc'][np.abs(spec1d_table[stds]['airmass'] - row['airmass']).argsort()[1]]
                spec1d_table.loc[row['filename']]['sensfunc'] = best_sens
        else:
            print("Could not find valid standard for sensitivity function, and no archived files found")
            #for filename in spec1d_table[arm]['filename']:
                #spec1d_table.loc[filename]['sensfunc'] = config

    # build fluxfile
    if do_I:
        spec1d_to_sensfunc = {row['filename']: row['sensfunc'] for row in spec1d_table if row['arm'] == 'I'}
        I_fluxfile = fluxing.build_fluxfile(spec1d_to_sensfunc, output_path_I, 'p200_ngps_I', I_user_config_lines)
    if do_R:
        spec1d_to_sensfunc = {row['filename']: row['sensfunc'] for row in spec1d_table if row['arm'] == 'R'}
        R_fluxfile = fluxing.build_fluxfile(spec1d_to_sensfunc, output_path_R, 'p200_ngps_R', R_user_config_lines)

    # flux data
    if do_I:
        fluxing.flux(I_fluxfile, output_path_I)
    if do_R:
        fluxing.flux(R_fluxfile, output_path_R)

    # coadd - intelligent coadding of multiple files
    # first make a column "coaddID" that is the same for frames to be coadded
    # TODO: when there are multiple exposures of an object, splice/output all of them
    #coaddIDs = []
    """
    if args.null_coadd:
        coaddIDs = range(len(spec1d_table))
    else:
        previous_row : Row = None
        S_PER_DAY = 24 * 60 * 60
        thresh = 15
        for i, row in enumerate(spec1d_table):
            if i == 0:
                coaddIDs.append(0)
            else:
                # if this is the same object as the last one
                # and they were taken consecutively
                if ((row['arm'] == previous_row['arm']) and
                    (row['object'] == previous_row['object']) and
                    ((row['mjd']*S_PER_DAY - previous_row['mjd']*S_PER_DAY
                        - previous_row['exptime']) < previous_row['exptime'])):
                    coaddIDs.append(coaddIDs[-1])
                else:
                    coaddIDs.append(coaddIDs[-1] + 1)
            previous_row = row

    spec1d_table.add_column(coaddIDs, name="coadd_id")

    # figure out where on detector likely target is
    spec1d_table.add_column(Column(name="spats", dtype=object, length=len(spec1d_table)))
    spec1d_table.add_column(Column(name="fracpos", dtype=object, length=len(spec1d_table)))
    all_spats = []
    all_fracpos = []
    # for each spec1d file
    for filename in spec1d_table['filename']:

        if "_NGPS_I_" in filename:
            path = os.path.join(output_path_I, 'Science', filename)
        elif "_NGPS_R_" in filename:
            path = os.path.join(output_path_R, 'Science', filename)
        with fits.open(path) as hdul:
            spats = []
            fracpos = []
            for i in range(1, len(hdul) - 1):
                # grab all of its extensions' spatial positions
                spats.append(int(hdul[i].name.split('-')[0].lstrip('SPAT')))
                fracpos.append(hdul[i].header['SPAT_FRACPOS'])
            spats.sort()
            fracpos.sort()
            all_spats.append(spats)
            all_fracpos.append(fracpos)
            spec1d_table.loc[filename]['spats'] = spats
            spec1d_table.loc[filename]['fracpos'] = fracpos
    # add to table???
    # this needs to be dtype object to allow for variable length lists
    spec1d_table.add_column(Column(name="coadds", dtype=object, length=len(spec1d_table)))
    spec1d_table.add_column([False]*len(all_spats), name="processed")

    ###########################################################################################################################################################
    # why does this set points to zero ? 

    #sn_smooth_npix=self.par['sn_smooth_npix'], wave_method=self.par['wave_method'],
    #dv=self.par['dv'], dwave=self.par['dwave'], dloglam=self.par['dloglam'],
    #wave_grid_min=self.par['wave_grid_min'], wave_grid_max=self.par['wave_grid_max'],
    #spec_samp_fact=self.par['spec_samp_fact'], ref_percentile=self.par['ref_percentile'],
    #maxiter_scale=self.par['maxiter_scale'], sigrej_scale=self.par['sigrej_scale'],
    #scale_method=self.par['scale_method'], sn_min_medscale=self.par['sn_min_medscale'],
    #sn_min_polyscale=self.par['sn_min_polyscale'], weight_method = self.par['weight_method'],
    #maxiter_reject=self.par['maxiter_reject'], lower=self.par['lower'], upper=self.par['upper'],
    #maxrej=self.par['maxrej'], sn_clip=self.par['sn_clip'], debug=self.debug, show=self.show)

    # coadd
    # iterate over coadd_ids
    coadd_to_spec1d = {}
    for coadd_id in set(coaddIDs):
        subtable = spec1d_table[spec1d_table['coadd_id'] == coadd_id]
        fname_spats = {row['filename']: row['spats'].copy() for row in subtable}
        grouped_spats_list = coadding.group_coadds(fname_spats)
        if all(subtable['arm'] == 'I'):
            coadds = coadding.coadd(grouped_spats_list, output_path_I, 'p200_ngps_I', I_user_config_lines)
        if all(subtable['arm'] == 'R'):
            coadds = coadding.coadd(grouped_spats_list, output_path_R, 'p200_ngps_R', R_user_config_lines)
        assert all(subtable['arm'] == 'I') or all(subtable['arm'] == 'R'),\
            "Something went wrong with coadding..."
        for row in subtable:
            spec1d_table.loc[row['filename']]['coadds'] = coadds
        for i, coadd in enumerate(coadds):
            coadd_to_spec1d[coadd] = list(zip(grouped_spats_list[i]['fnames'], grouped_spats_list[i]['spats']))

    ###########################################################################################################################################################


    # REMOVED CODE HERE TO SKIP TELLURIC ###############################

    # current splicing - make sure spatial fraction is similar on blue/red
    # TODO: handle multiple observations of same target throughout night with null coadding
    # splice data
    splicing_dict = {}
    R_mask = spec1d_table['arm'] == 'R'
    I_mask = spec1d_table['arm'] == 'I'

    os.makedirs(os.path.join(args.output_path, 'spliced'), exist_ok=True)

    def get_std_trace(std_path: str) -> float:
        # Return the fractional position of the highest SNR trace
        max_sn = -1
        max_fracpos = -1
        with fits.open(std_path) as hdul:
            # loop through trace hdus
            for hdu in hdul:
                if not 'SPAT' in hdu.name:
                    continue

                # look at s/n
                if 'OPT_COUNTS' in hdu.data.dtype.names:
                    this_sn = np.nanmedian(hdu.data['OPT_COUNTS']/hdu.data['OPT_COUNTS_SIG'])
                elif 'BOX_COUNTS' in hdu.data.dtype.names:
                    this_sn = np.nanmedian(hdu.data['BOX_COUNTS']/hdu.data['BOX_COUNTS_SIG'])
                else:
                    this_sn = -1

                if this_sn > max_sn:
                    max_sn = this_sn
                    max_fracpos = hdu.header['SPAT_FRACPOS']

        if max_fracpos == -1:
            raise Exception(f"Error! No HDUs in {os.path.basename(std_path)} have median S/N > 0.")
        return max_fracpos

    ## Need to find red + blue fracpos for standards
    # hopefully standards only have one star each?
    # or should i actually try to do matching there
    stds = spec1d_table['frametype'] == 'standard'
    if do_I or do_R:
        FRACPOS_SUM = 1.0
        FRACPOS_TOL = 0.05
        if do_I and do_R:
            # real matching + splicing
            std_fracpos_sums = []
            if (stds & R_mask).any() and (stds & I_mask).any():
                for row in spec1d_table[stds]:
                    if row['arm'] == 'I':
                    # find closest mjd frame of other arm
                        if not row['processed']:
                            other_arm = spec1d_table['arm'] != row['arm']
                            corresponding_row = spec1d_table[other_arm & stds][np.abs(spec1d_table[other_arm & stds]['mjd'] - row['mjd']).argmin()]
                            this_path = os.path.join(output_path_I, 'Science', row['filename'])
                            corresponding_path = os.path.join(output_path_R, 'Science', corresponding_row['filename'])
                            std_fracpos_sums.append(get_std_trace(this_path) + get_std_trace(corresponding_path))
                            spec1d_table.loc[row['filename']]['processed'] = True
                            spec1d_table.loc[corresponding_row['filename']]['processed'] = True
                    FRACPOS_SUM = np.mean(std_fracpos_sums)
                    FRACPOS_TOL = FRACPOS_SUM * .025

        # setup splicing dict
        splicing_dict = {}
        # for each target
        for row in spec1d_table:
            target = row['object']
            arm = row['arm']
            # for each of its fracpos
            for i, fracpos in enumerate(row['fracpos']):
                coadd = row['coadds'][i]
                targ_dict = splicing_dict.get(target)
                # normalize fracpos to red
                if do_I and do_R and arm == 'R':
                    fracpos = FRACPOS_SUM - fracpos
                # if it's not in the dict
                if targ_dict is None:
                    # put it in the dict
                    splicing_dict[target] = {fracpos: {
                        arm: {
                            'spec1ds': coadd_to_spec1d[coadd],
                            'coadd': coadd
                        }
                    }}
                # else
                else:
                    close_enough = False
                    # for each existing fracpos
                    for fracpos_existing in list(targ_dict):
                        # if its close enough
                        if abs(fracpos_existing - fracpos) < FRACPOS_TOL:
                            # put it in the dict
                            splicing_dict[target][fracpos_existing][arm] = {
                                'spec1ds': coadd_to_spec1d[coadd],
                                'coadd': coadd
                            }
                            close_enough = True
                            break
                    if not close_enough:
                        # If this fracpos isn't close enough to any others
                        splicing_dict[target][fracpos] = {arm: {
                            'spec1ds': coadd_to_spec1d[coadd],
                            'coadd': coadd
                        }}
        # And now, actually splice!
        
        splicing.splice(splicing_dict, args.splicing_interpolate_gaps, args.root, output_path_I, output_path_R, os.path.join(args.output_path, 'spliced'))"
    """

    print('Elapsed time: {0} seconds'.format(time.perf_counter() - t))
