import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import KDTree

class StarTracker:
    def __init__(self, input_folder, output_pdf="tracked_stars.pdf"):
        self.input_folder = input_folder
        self.output_pdf = output_pdf
        self.files = self.get_sorted_files()
        self.algol_positions = []
        self.ref_positions = []

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

            algol_pos = self.find_star(image_data, x_guess, y_guess)
            self.algol_positions.append(algol_pos)

    def track_reference_star(self):
        """Track a reference star using KDTree matching."""
        first_file = os.path.join(self.input_folder, self.files[0])
        with fits.open(first_file) as hdul:
            image_data = hdul[0].data

        # User selects reference star in first frame
        plt.figure(figsize=(8, 6))
        self.display_image_with_zscale(image_data, title="Select Reference Star (First Frame)")
        ref_pos = plt.ginput(1, timeout=0)[0]
        plt.close()

        ref_pos = self.find_star(image_data, *ref_pos)
        self.ref_positions.append(ref_pos)

        # Track reference star across frames using KDTree
        for i, file in enumerate(self.files[1:], start=1):
            file_path = os.path.join(self.input_folder, file)
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data

            if self.ref_positions[-1] is None:
                self.ref_positions.append(None)
                continue

            # Detect stars in the current frame
            detected_stars = self.detect_stars(image_data)
            if len(detected_stars) == 0:
                self.ref_positions.append(None)
                continue

            # Match closest star to the last known position using KDTree
            kdtree = KDTree(detected_stars)
            _, idx = kdtree.query(self.ref_positions[-1])
            self.ref_positions.append(tuple(detected_stars[idx]))

    def save_pdf(self):
        """Save tracked stars as a multi-page PDF."""
        with PdfPages(self.output_pdf) as pdf:
            for i, file in enumerate(self.files):
                file_path = os.path.join(self.input_folder, file)
                with fits.open(file_path) as hdul:
                    image_data = hdul[0].data

                timestamp = file.split("_")[2]
                exposure_time = hdul[0].header["EXPOSURE"]

                plt.figure(figsize=(8, 6))
                self.display_image_with_zscale(image_data, title=f"Frame {i+1}: {timestamp}\nExposure: {exposure_time}s")

                if self.algol_positions[i] is not None:
                    plt.scatter(*self.algol_positions[i], color='blue', marker='x', s=100, label="Algol")

                if i < len(self.ref_positions) and self.ref_positions[i] is not None:
                    plt.scatter(*self.ref_positions[i], color='red', marker='x', s=10, alpha=0.5, label="Reference Star")

                plt.legend()
                pdf.savefig()
                plt.close()

        print(f"Saved tracked positions to {self.output_pdf}")

    def run(self):
        self.track_algol()
        self.track_reference_star()
        self.save_pdf()

# Example Usage
input_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/calibrated_frames"
tracker = StarTracker(input_folder)
tracker.run()
