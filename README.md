<<<<<<< HEAD
# NGPS_PIPE
=======

## Description
NGPS_PIPE is a Data Reduction Pipeline for Palomar's NGPS.
It is built on top of [PypeIt](https://github.com/pypeit/PypeIt).
NGPS_PIPE automates the reduction, fluxing, telluric correction, and combining of the R and I sides of one night's
data.
It adds several GUIs to allow for easier control of your reduction:
- select which data to reduce, and verify the correctness of your FITS headers in an editable table GUI
- manually place traces for a sort of manually "forced" spectroscopy with the `-m` option

## Usage
```shell_session
$ ngps_reduce -r /path/to/data/ /path/to/data/redux
    [-a {I,R}] [-i] [-m] [--debug] [-j N] [-p PARAMETER_FILE] [-t] [-c]
```
>>>>>>> 4d5c2a7 (Initial commit of NGPS_Pipe)
