import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_pdf import PdfPages

def save_frames_to_pdf(input_folder, output_pdf="all_frames.pdf"):
    """Saves all frames as individual pages in a PDF."""
    files = sorted([f for f in os.listdir(input_folder) if f.endswith("_df.fits") and f.startswith("algol")])

    pdf_path = os.path.join(input_folder, output_pdf)
    print(f"Saving frames to {pdf_path}...")

    zscale = ZScaleInterval()

    with PdfPages(pdf_path) as pdf:
        for file in files:
            file_path = os.path.join(input_folder, file)
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header

                timestamp = header.get("DATE-OBS", "Unknown Time")
                exposure = header.get("EXPOSURE", "Unknown Exposure")

            vmin, vmax = zscale.get_limits(image_data)

            plt.figure(figsize=(8, 6))
            plt.imshow(image_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            plt.colorbar(label="Pixel Value")
            plt.title(f"Timestamp: {timestamp}\nExposure: {exposure} sec")

            pdf.savefig()
            plt.close()

    print("PDF saved successfully!")

# Example usage
input_folder = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/calibrated_frames"
save_frames_to_pdf(input_folder)
