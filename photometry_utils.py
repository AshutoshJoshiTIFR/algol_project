import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import KDTree
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.io import fits


class StarTracker:
    def __init__(self, input_folder, output_pdf="tracked_stars.pdf"):
        self.input_folder = input_folder
        self.output_pdf = output_pdf
        self.files = self.get_sorted_files()
        self.algol_positions = []
        self.ref_positions = []  # List of lists for multiple reference stars
        self.timestamps = []

    def get_sorted_files(self):
        """Sort files by timestamp."""
        files = [f for f in os.listdir(self.input_folder) if f.endswith("_df.fits") and f.startswith("algol")]
        files.sort(key=lambda x: x.split("_")[2])  # Sorting by timestamp
        return files

    def display_image_with_zscale(self, image_data, title="Select Star"):
        """Display an image with ZScale normalization."""
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_data)

        plt.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label="Pixel Value")
        plt.title(title)

    def detect_stars(self, image_data):
        """Detect stars in the entire image using DAOStarFinder."""
        daofind = DAOStarFinder(fwhm=5.0, threshold=1.5 * np.std(image_data))
        sources = daofind(image_data)

        if sources is None or len(sources) == 0:
            return np.array([])

        return np.array([sources["xcentroid"], sources["ycentroid"]]).T  # (N, 2) array of star positions

    def find_star(self, image_data, x_guess, y_guess, search_box=20):
        """Find the nearest star to the given position."""
        ny, nx = image_data.shape
        x_min, x_max = max(0, int(x_guess - search_box)), min(nx, int(x_guess + search_box))
        y_min, y_max = max(0, int(y_guess - search_box)), min(ny, int(y_guess + search_box))

        cropped_image = image_data[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0:
            return None

        cropped_image = np.nan_to_num(cropped_image)
        stars = self.detect_stars(cropped_image)

        if len(stars) == 0:
            return None

        # Convert local coordinates to global
        stars[:, 0] += x_min
        stars[:, 1] += y_min

        # Find the nearest detected star to the guess
        kdtree = KDTree(stars)
        _, idx = kdtree.query([x_guess, y_guess])
        return tuple(stars[idx])

    def track_algol(self):
        """Track Algol using PHDLOCK header positions as initial guess."""
        for file in self.files:
            file_path = os.path.join(self.input_folder, file)
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                x_guess, y_guess = header["PHDLOCKX"], header["PHDLOCKY"]
                timestamp = header["DATE-OBS"]  # Get timestamp from FITS header

            algol_pos = self.find_star(image_data, x_guess, y_guess)
            self.algol_positions.append(algol_pos)
            self.timestamps.append(timestamp)

    def track_reference_stars(self):
        """Prompt user to select multiple reference stars and track them across frames."""
        first_file = os.path.join(self.input_folder, self.files[0])
        with fits.open(first_file) as hdul:
            image_data = hdul[0].data

        # Ask user for the number of reference stars
        num_ref_stars = int(input("Enter the number of reference stars: "))

        # User selects reference stars in the first frame
        plt.figure(figsize=(8, 6))
        self.display_image_with_zscale(image_data, title="Select Reference Stars (First Frame)")
        ref_positions = plt.ginput(num_ref_stars, timeout=0)
        plt.close()

        # Store refined positions after peak detection
        self.ref_positions = [[] for _ in range(num_ref_stars)]
        for i, pos in enumerate(ref_positions):
            refined_pos = self.find_star(image_data, *pos)
            self.ref_positions[i].append(refined_pos)

        # Track reference stars across all frames
        for file_idx, file in enumerate(self.files[1:], start=1):
            file_path = os.path.join(self.input_folder, file)
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data

            for i in range(num_ref_stars):
                if self.ref_positions[i][-1] is None:
                    self.ref_positions[i].append(None)
                    continue

                detected_stars = self.detect_stars(image_data)
                if len(detected_stars) == 0:
                    self.ref_positions[i].append(None)
                    continue

                kdtree = KDTree(detected_stars)
                _, idx = kdtree.query(self.ref_positions[i][-1])
                self.ref_positions[i].append(tuple(detected_stars[idx]))

    def save_pdf(self):
        """Save tracked stars as a multi-page PDF."""
        with PdfPages(self.output_pdf) as pdf:
            for i, file in enumerate(self.files):
                file_path = os.path.join(self.input_folder, file)
                with fits.open(file_path) as hdul:
                    image_data = hdul[0].data

                timestamp = self.timestamps[i]
                exposure_time = hdul[0].header["EXPOSURE"]

                plt.figure(figsize=(8, 6))
                self.display_image_with_zscale(image_data, title=f"Frame {i+1}: {timestamp}\nExposure: {exposure_time}s")

                if self.algol_positions[i] is not None:
                    plt.scatter(*self.algol_positions[i], color='blue', marker='x', s=100, label="Algol")

                for j, ref_pos_list in enumerate(self.ref_positions):
                    if i < len(ref_pos_list) and ref_pos_list[i] is not None:
                        plt.scatter(*ref_pos_list[i], color='red', marker='o', s=50, alpha=0.5, label=f"Ref Star {j+1}")

                plt.legend()
                pdf.savefig()
                plt.close()

        print(f"Saved tracked positions to {self.output_pdf}")

    def run(self):
        self.track_algol()
        self.track_reference_stars()
        self.save_pdf()


class Photometry:
    def __init__(self, algol_positions, ref_positions, timestamps, files):
        self.algol_positions = algol_positions
        self.ref_positions = ref_positions
        self.timestamps = timestamps
        self.files = files
        self.aperture_width = self.find_aperture_width()
        self.algol_flux = None
        self.ref_flux = None
        self.algol_rel_flux = None

    def find_aperture_width(self):
        """
        Fits a 2d gaussian to algol profile and finds its aperture width.
        Can be done for first file only and fixed thereafter.
        Same aperture is to be used for other reference stars. Note that 
        the data is noisy.
        """
        pass

    
    def calculate_flux(self, annulus_inner_radius, annulus_outer_radius):
        """
        This function should calulate flux in each frame for Algol, and
        reference stars and stores it in some dictionary in the format
        {timestamp : (flux, variance)}.
        Background subtraction is to be done.
        Errors are supposed to be poissonian and independent of each other.
        Errors(standard deviation) should be added in quadratures when variables
        are added/subtracted.
        Check Barlow before writing this method.

        Each frame should be normalized by its exposure time before doing
        any of the above.
        """
        pass


    def relative_photmetry(self):
        """
        This function calculates the relative flux of algol with respect to 
        each of the reference stars.
        My recommended format is 
        {ref 1:
        {timestamp:flux}
        ref 2:
        {timestamp:flux}
        .
        .
        ref n:
        {timestamp:flux}}
        """
        pass


    def save_pdf(self, output_file="algol_lightcurves.pdf"):
        """
        This function plots the relative flux of algol with respect
        to each reference star. Plot errorbars also. And saves it in 
        a pdf file, one plot per page.
        """

