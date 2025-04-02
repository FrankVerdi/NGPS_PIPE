import os
from astropy.io import fits
from pathlib import Path
import argparse

def process_fits_files(input_dir, output_dir):
    """
    Process FITS files in a directory to ensure correct SPEC_ID values in extensions 1 and 2.

    Args:
        input_dir (str): Path to the directory containing the input FITS files.
        output_dir (str): Path to the directory to save corrected FITS files.
    """
    print("Fixing image order in raw fits files...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all FITS files in the input directory
    for fits_file in input_dir.glob("*.fits"):
        with fits.open(fits_file, mode='readonly') as hdul:
            # Ensure the file has at least 3 HDUs (Primary + extensions 1 and 2)
            if len(hdul) < 3:
                print(f"Skipping {fits_file.name}: Not enough extensions.")
                continue

            # Extract SPEC_ID values
            spec_id_1 = hdul[1].header.get('SPEC_ID', None)
            spec_id_2 = hdul[2].header.get('SPEC_ID', None)

            # Check if swap is needed
            if spec_id_1 == 'I' and spec_id_2 == 'R':
                print(f"Swapping extensions in {fits_file.name}")

                # Swap the data and headers
                hdul[1].data, hdul[2].data = hdul[2].data, hdul[1].data
                hdul[1].header, hdul[2].header = hdul[2].header, hdul[1].header

            # Save the corrected FITS file to the output directory
            corrected_file_path = output_dir / fits_file.name
            hdul.writeto(corrected_file_path, overwrite=True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process FITS files to correct SPEC_ID headers.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing FITS files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for corrected FITS files.")
    args = parser.parse_args()

    # Run the processing function with the provided arguments
    process_fits_files(args.input_dir, args.output_dir)

