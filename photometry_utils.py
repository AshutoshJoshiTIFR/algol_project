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
    def __init__(self, input_folder, algol_positions, reference_positions, timestamps, aperture_radius=5, annulus_radius=(10, 15), output_pdf="light_curves.pdf"):
        self.input_folder = input_folder
        self.aperture_radius = aperture_radius
        self.annulus_radius = annulus_radius
        self.output_pdf = output_pdf
        self.algol_positions = algol_positions
        self.reference_positions = reference_positions
        self.timestamps = timestamps

    def get_sorted_files(self):
        """Sort files by timestamp."""
        files = [f for f in os.listdir(self.input_folder) if f.endswith("_df.fits") and f.startswith("algol")]
        files.sort(key=lambda x: x.split("_")[2])  # Sorting by timestamp
        return files

    def perform_photometry(self, algol_positions, reference_positions, timestamps):
        """
        Perform aperture photometry for Algol and reference stars.
        Return fluxes and errors for Algol and reference stars.
        """
        algol_positions = self.algol_positions
        reference_positions = self.reference_positions
        timestamps = self.timestamps
        algol_fluxes = []
        algol_errors = []
        ref_fluxes = {i: [] for i in range(len(reference_positions))}
        ref_errors = {i: [] for i in range(len(reference_positions))}

        sorted_files = self.get_sorted_files()

        for i, timestamp in enumerate(timestamps):
            # Find the matching filename for the timestamp
            matching_file = next((f for f in sorted_files if timestamp in f), None)
            if matching_file is None:
                print(f"Warning: No file found for timestamp {timestamp}")
                continue

            file_path = os.path.join(self.input_folder, matching_file)

            # Open the FITS file
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                exposure_time = hdul[0].header["EXPOSURE"]

            # Perform aperture photometry for Algol
            algol_flux, algol_error = self.calculate_flux(image_data, algol_positions[i])
            algol_flux /= exposure_time  # Normalize by exposure time

            # Append Algol's flux and error
            algol_fluxes.append(algol_flux)
            algol_errors.append(algol_error)

            # Perform aperture photometry for reference stars
            for j, ref_pos in enumerate(reference_positions):
                ref_flux, ref_error = self.calculate_flux(image_data, ref_pos[i])
                ref_flux /= exposure_time  # Normalize by exposure time

                # Store flux and errors for reference stars
                ref_fluxes[j].append(ref_flux)
                ref_errors[j].append(ref_error)

        return algol_fluxes, algol_errors, ref_fluxes, ref_errors

    def calculate_flux(self, image_data, position):
        """
        Calculate the flux and error for a given star position using aperture photometry.
        Propagates error from the star signal and the background noise in the annulus.
        """
        if position is None:
            return 0, 0  # No flux if no position is found

        y, x = position
        aperture = CircularAperture((x, y), r=self.aperture_radius)
        annulus = CircularAnnulus((x, y), r_in=self.annulus_radius[0], r_out=self.annulus_radius[1])

        # Mask the image for aperture and annulus
        aperture_mask = aperture.to_mask(method='center')
        annulus_mask = annulus.to_mask(method='center')

        # Get the flux and background in the annulus
        aperture_flux = aperture_mask.multiply(image_data).sum()
        annulus_flux = annulus_mask.multiply(image_data).sum()

        # Estimate background noise using the annulus
        background = annulus_flux / annulus_mask.area
        annulus_std = np.std(image_data[annulus_mask.data > 0])

        # Subtract background from aperture flux
        flux = aperture_flux - background * aperture_mask.area

        # Calculate the error
        error = np.sqrt(flux + (annulus_std ** 2) * aperture_mask.area)

        return flux, error

    def save_light_curve_pdf(self, algol_fluxes, algol_errors, ref_fluxes, ref_errors):
        """Save light curves as a PDF."""
        with PdfPages(self.output_pdf) as pdf:
            # Create plot for Algol
            plt.figure(figsize=(8, 6))
            plt.errorbar(range(len(algol_fluxes)), algol_fluxes, yerr=algol_errors, fmt='o', label="Algol")
            plt.title("Algol Light Curve")
            plt.xlabel("Frame")
            plt.ylabel("Flux")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Create plots for reference stars
            for i in range(len(ref_fluxes)):
                plt.figure(figsize=(8, 6))
                plt.errorbar(range(len(ref_fluxes[i])), ref_fluxes[i], yerr=ref_errors[i], fmt='o', label=f"Ref Star {i+1}")
                plt.title(f"Reference Star {i+1} Light Curve")
                plt.xlabel("Frame")
                plt.ylabel("Flux")
                plt.legend()
                pdf.savefig()
                plt.close()

        print(f"Saved light curves to {self.output_pdf}")

    def run(self):
        algol_fluxes, algol_errors, ref_fluxes, ref_errors = self.perform_photometry(self.algol_positions, self.reference_positions, self.timestamps)
        self.save_light_curve_pdf(algol_fluxes, algol_errors, ref_fluxes, ref_errors)


if __name__ == "__main__":
    input_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/calibrated_frames"
    tracker = StarTracker(input_folder)
    tracker.run()

    # For photometry, you can initialize with tracked positions
    photometry = Photometry(input_folder, tracker.algol_positions, tracker.ref_positions, tracker.timestamps)
    photometry.run()
