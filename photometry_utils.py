import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import KDTree
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import fit_2dgaussian
from astropy.io import fits
from tqdm import tqdm


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
        for file in tqdm(self.files, desc="Tracking Algol."):
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
        for file_idx, file in tqdm(enumerate(self.files[1:], start=1), desc="Tracking reference stars."):
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
            for i, file in tqdm(enumerate(self.files), desc="Saving tracked_stars.pdf"):
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
        print("Begin Tracking.")
        print("--------------")
        self.track_algol()
        self.track_reference_stars()
        self.save_pdf()
        print("Tracking Done.")
        print("--------------")


class Photometry:
    def __init__(self, algol_positions, ref_positions, timestamps, files, input_folder):
        self.algol_positions = algol_positions
        self.ref_positions = ref_positions
        self.timestamps = timestamps
        self.files = files
        self.input_folder = input_folder
        #self.aperture_radius = self.find_aperture_radius()
        
        self.aperture_radius = 5

        self.algol_flux = {}  # {timestamp: (flux, variance)}
        self.ref_flux = [{} for _ in range(len(ref_positions))]  # [{timestamp: (flux, variance)}, ...]
        self.algol_rel_flux = {}

    '''
    def gaussian_2d(self, xy, A, x0, y0, sigma_x, sigma_y, C):
        """2D Gaussian function"""
        x, y = xy
        return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))) + C
    

    def find_aperture_radius(self):
        """
        Fits a 2D Gaussian to Algol's profile and finds its aperture width.
        Uses the first FITS file and applies the same aperture for all stars.
        """
        from scipy.optimize import curve_fit
        file_path = os.path.join(self.input_folder, self.files[0])
        with fits.open(file_path) as hdul:
            image_data = hdul[0].data

        x0, y0 = self.algol_positions[0]

        # Define a small cutout region around Algol
        cutout_size = 15
        x_min, x_max = int(x0 - cutout_size), int(x0 + cutout_size)
        y_min, y_max = int(y0 - cutout_size), int(y0 + cutout_size)
        cutout = image_data[y_min:y_max, x_min:x_max]

        # Generate coordinate grids
        y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
        x_data, y_data = x_indices.ravel(), y_indices.ravel()
        z_data = cutout.ravel()

        # Initial guesses
        A_guess = np.max(cutout) - np.median(cutout)
        sigma_guess = cutout_size / 4
        background_guess = np.median(cutout)
        initial_guess = [A_guess, x0, y0, sigma_guess, sigma_guess, background_guess]

        # Fit the Gaussian
        try:
            popt, _ = curve_fit(self.gaussian_2d, (x_data, y_data), z_data, p0=initial_guess)
            _, _, _, sigma_x, sigma_y, _ = popt  # Extract Gaussian sigmas
        except RuntimeError:
            print("Gaussian fit failed! Using fallback aperture.")
            return cutout_size / 4  # Fallback aperture

        # Convert stddev to FWHM and take the average for a circular aperture
        fwhm_x = 2.355 * sigma_x
        fwhm_y = 2.355 * sigma_y
        circular_fwhm = (fwhm_x + fwhm_y) / 2

        print(circular_fwhm)

        return circular_fwhm / 2

    
    def find_aperture_radius(self):
        """
        Fits a 2d gaussian to algol profile and finds its aperture width.
        Can be done for first file only and fixed thereafter.
        Same aperture is to be used for other reference stars. Note that 
        the data is noisy.
        """
        file_path = os.path.join(self.input_folder, self.files[0])

        with fits.open(file_path) as hdul:
            image_data = hdul[0].data

        x0, y0 = self.algol_positions[0]

        # Define a small cutout region around Algol
        cutout_size = 15
        x_min, x_max = int(x0 - cutout_size), int(x0 + cutout_size)
        y_min, y_max = int(y0 - cutout_size), int(y0 + cutout_size)
        cutout = image_data[y_min:y_max, x_min:x_max]

        # Fit a 2D Gaussian
        fit = fit_2dgaussian(cutout, xypos=(x0, y0))
        print("Circular Aperture Radius = ", fit.fwhm)
        return fit.fwhm
        # Convert stddev to FWHM and take the average for a circular aperture
        #fwhm_x = 2.355 * fit.x_stddev
        #fwhm_y = 2.355 * fit.y_stddev
        #circular_fwhm = (fwhm_x + fwhm_y) / 2

        
        #return circular_fwhm / 2
    '''
    
    def calculate_flux(self, annulus_inner_radius, annulus_outer_radius):
        """
        Calculates flux for Algol and reference stars in each frame.
        Background subtraction is applied.
        Errors are assumed Poissonian and independent.
        Each frame is normalized by its exposure time before calculations.
        """
        for i, file in tqdm(enumerate(self.files), desc="Calculating Flux"):
            file_path = os.path.join(self.input_folder, file)
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                exposure_time = header["EXPOSURE"]

            timestamp = self.timestamps[i]

            # Process Algol separately
            algol_aperture = CircularAperture(self.algol_positions[i], r=self.aperture_radius)
            algol_annulus = CircularAnnulus(self.algol_positions[i], r_in=annulus_inner_radius, r_out=annulus_outer_radius)

            phot_table_algol = aperture_photometry(image_data, algol_aperture)
            annulus_table_algol = aperture_photometry(image_data, algol_annulus)

            # Compute background for Algol
            bkg_mean_algol = annulus_table_algol["aperture_sum"] / algol_annulus.area
            bkg_total_algol = bkg_mean_algol * algol_aperture.area

            # Compute flux and variance for Algol
            flux_algol = (phot_table_algol["aperture_sum"][0] - bkg_total_algol) / exposure_time
            var_algol = (phot_table_algol["aperture_sum"][0] + bkg_total_algol) / (exposure_time**2)

            self.algol_flux[timestamp] = (flux_algol, var_algol)

            # Process each reference star separately
            for j, ref_pos in enumerate(self.ref_positions):
                ref_aperture = CircularAperture(ref_pos[i], r=self.aperture_radius)
                ref_annulus = CircularAnnulus(ref_pos[i], r_in=annulus_inner_radius, r_out=annulus_outer_radius)

                phot_table_ref = aperture_photometry(image_data, ref_aperture)
                annulus_table_ref = aperture_photometry(image_data, ref_annulus)

                # Compute background for reference star
                bkg_mean_ref = annulus_table_ref["aperture_sum"] / ref_annulus.area
                bkg_total_ref = bkg_mean_ref * ref_aperture.area

                # Compute flux and variance for reference star
                flux_ref = (phot_table_ref["aperture_sum"][0] - bkg_total_ref) / exposure_time
                var_ref = (phot_table_ref["aperture_sum"][0] + bkg_total_ref) / (exposure_time**2)

                self.ref_flux[j][timestamp] = (flux_ref, var_ref)




    def relative_photmetry(self):
        """
        This function calculates the relative flux of algol with respect to 
        each of the reference stars.
        My recommended format is 
        {ref 1:
        {timestamp:(flux, sigma)}
        ref 2:
        {timestamp:(flux, sigma)}
        .
        .
        ref n:
        {timestamp:(flux, sigma)}}
        """
        for j in range(len(self.ref_positions)):
            self.algol_rel_flux[f"ref_{j+1}"] = {}
            for timestamp in self.algol_flux.keys():
                flux_algol, var_algol = self.algol_flux[timestamp]
                if timestamp in self.ref_flux[j]:
                    flux_ref, var_ref = self.ref_flux[j][timestamp]
                    rel_flux = flux_algol / flux_ref
                    rel_var = rel_flux**2 * (var_algol / flux_algol**2 + var_ref / flux_ref**2)  # Error propagation
                    self.algol_rel_flux[f"ref_{j+1}"][timestamp] = (rel_flux, np.sqrt(rel_var))
        

    def save_pdf(self, output_file="algol_lightcurves.pdf"):
        """
        This function plots the relative flux of algol with respect
        to each reference star. Plot errorbars also. And saves it in 
        a pdf file, one plot per page.
        """
        with PdfPages(output_file) as pdf:
            for ref_name, flux_dict in tqdm(self.algol_rel_flux.items(), desc="Saving Lightcurves."):
                timestamps = self.timestamps
                time_labels = [ts.split("T")[1][:-7] for ts in timestamps]

                flux_values = np.array([flux_dict[t][0] for t in timestamps]).flatten()
                flux_errors = np.array([flux_dict[t][1] for t in timestamps]).flatten()
                
                plt.figure(figsize=(8, 6))
                plt.errorbar(time_labels, flux_values, yerr=flux_errors, color="k", ecolor="red", fmt='o', capsize=3)
                plt.xlabel("Time(UTC)")
                plt.ylabel("Relative Flux (Algol / " + ref_name + ")")
                plt.title(f"Algol Light Curve (Relative to {ref_name})")
                plt.grid(True)
                plt.ylim(0, 255)
                
                step = max(1, len(time_labels) // 10)  # Show ~10 ticks max
                plt.xticks(time_labels[::step], rotation=45)
                pdf.savefig()
                plt.close()

        print(f"Saved light curves to {output_file}")

    def run(self):
        print("Begin Photometry")
        print("----------------")
        self.calculate_flux(2*self.aperture_radius, 3*self.aperture_radius)
        self.relative_photmetry()
        self.save_pdf()
        print("Photometry Done")
        print("----------------")


input_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/calibrated_frames/"

tracker = StarTracker(input_folder)
tracker.run()

photometry = Photometry(tracker.algol_positions, tracker.ref_positions, tracker.timestamps, tracker.files, input_folder)
photometry.run()