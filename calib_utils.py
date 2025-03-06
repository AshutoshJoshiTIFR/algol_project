import os
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class DataReduction:
    def __init__(self, input_folder, output_folder):
        """
        Initializes the DataReduction class.
        
        Parameters:
        - input_folder (str): Directory containing raw FITS files
        - output_folder (str): Directory to save processed FITS files
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processed_data = {}  # Stores dark-subtracted science frames
        self.flat_field = None     # Processed flat frame
        self.dark_frames = {}      # Dictionary to store darks by exposure time

        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Categorize files
        self.science_files = {'algol': [], 'capella': []}
        self.dark_files = []
        self.flat_file = None

        self._categorize_files()

    def _categorize_files(self):
        """ Categorizes files into Algol, Capella, Dark, and Flat. """
        for filename in os.listdir(self.input_folder):
            if not filename.endswith(".fits"):
                continue
            
            filepath = os.path.join(self.input_folder, filename)
            parts = filename.split('_')

            if len(parts) < 3:
                continue  # Skip files that don't match expected naming

            event_type = parts[0]  # "algol", "capella", "flat", or "dark"
            exposure = float(parts[1])

            if event_type == "flat":
                self.flat_file = filepath
            elif event_type == "capella":
                self.science_files['capella'].append((filepath, exposure))
            elif event_type == "algol":
                self.science_files['algol'].append((filepath, exposure))
            elif event_type.startswith("dark"):
                self.dark_files.append((filepath, exposure))

        logging.info(f"Categorized {len(self.science_files['algol'])} Algol, {len(self.science_files['capella'])} Capella, {len(self.dark_files)} Dark, and 1 Flat file.")

    @staticmethod
    def smooth_image(image, sigma=50):
        """ Applies a high-pass filter to the flat frame. """
        smoothed = ndimage.gaussian_filter(image, sigma=sigma)
        return smoothed

    def process_flat(self):
        """ Loads, high-pass filters, and normalizes the flat frame. """
        if not self.flat_file:
            logging.warning("No flat frame found!")
            return

        with fits.open(self.flat_file) as hdu:
            flat_data = hdu[0].data.astype(np.float64)
            flat_header = hdu[0].header

        # Apply high-pass filter
        smoothed = self.smooth_image(flat_data)
        processed_flat = flat_data / smoothed  # Normalize

        # Save the processed flat
        flat_filename = os.path.join(self.output_folder, "flat_processed.fits")
        fits.writeto(flat_filename, processed_flat, flat_header, overwrite=True)
        self.flat_field = processed_flat
        logging.info(f"Processed flat frame saved as {flat_filename}")

    def subtract_dark(self):
        """ Subtracts dark frames from corresponding science frames. """
        # Organize dark frames by exposure time
        for dark_path, exposure in self.dark_files:
            with fits.open(dark_path) as hdu:
                self.dark_frames[exposure] = hdu[0].data.astype(np.float64)

        # Subtract dark from science images
        for category, files in self.science_files.items():
            for sci_path, exposure in files:
                with fits.open(sci_path) as hdu:
                    sci_data = hdu[0].data.astype(np.float64)
                    sci_header = hdu[0].header  # Keep header

                if exposure in self.dark_frames:
                    dark_data = self.dark_frames[exposure]
                    dark_subtracted = sci_data - dark_data
                else:
                    logging.warning(f"No dark frame for {exposure}s exposure. Skipping dark subtraction.")
                    dark_subtracted = sci_data  # Use raw science frame if no matching dark

                # Store result for flat correction
                self.processed_data[sci_path] = (dark_subtracted, sci_header)

                # Save dark-subtracted file
                new_filename = os.path.basename(sci_path).replace(".fits", "_d.fits")
                dark_filename = os.path.join(self.output_folder, new_filename)
                fits.writeto(dark_filename, dark_subtracted, sci_header, overwrite=True)
                logging.info(f"Dark-subtracted file saved: {dark_filename}")

    def apply_flat_correction(self):
        """ Applies flat field correction to dark-subtracted images. """
        if self.flat_field is None:
            logging.warning("Flat frame not processed. Skipping flat correction.")
            return

        for filename, (dark_subtracted, sci_header) in self.processed_data.items():
            flat_corrected = dark_subtracted / self.flat_field

            # Save flat-corrected file
            new_filename = os.path.basename(filename).replace(".fits", "_df.fits")
            flat_filename = os.path.join(self.output_folder, new_filename)
            fits.writeto(flat_filename, flat_corrected, sci_header, overwrite=True)
            logging.info(f"Flat-corrected file saved: {flat_filename}")

    def run_pipeline(self):
        """ Runs the entire calibration pipeline. """
        logging.info("Starting data reduction pipeline...")
        self.process_flat()
        self.subtract_dark()
        self.apply_flat_correction()
        logging.info("Data reduction complete.")


input_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/Reduction1"
output_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/calibrated_frames"

reduction = DataReduction(input_folder, output_folder)
reduction.run_pipeline()
