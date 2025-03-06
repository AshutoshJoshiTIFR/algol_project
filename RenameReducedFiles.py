import os
import glob

def rename_fits_files(reduction1_path):
    """
    Renames FITS files in the Reduction1 directory based on event numbers.

    Parameters:
    - reduction1_path (str): Path to the Reduction1 directory.
    """

    # Mapping of event numbers to new names
    event_mapping = {
        "086": "flat",
        "074": "capella"
    }

    # Process event numbers 001 to 073 as "algol"
    for i in range(1, 74):
        event_mapping[f"{i:03d}"] = "algol"

    # Process event numbers 075 to 085 as "dark"
    for i in range(75, 86):
        event_mapping[f"{i:03d}"] = "dark"

    # Find all FITS files in the directory
    fits_files = glob.glob(os.path.join(reduction1_path, "event*.fits"))

    for file in fits_files:
        filename = os.path.basename(file)

        # Extract the event number (first 8 characters after 'event')
        if filename.startswith("event") and "_" in filename:
            event_num = filename[5:8]  # Extract event number (e.g., "073" from "event073_...")
            
            if event_num in event_mapping:
                new_name = filename.replace(f"event{event_num}", event_mapping[event_num])
                new_path = os.path.join(reduction1_path, new_name)

                try:
                    os.rename(file, new_path)
                    print(f"Renamed: {filename} --> {new_name}")
                except Exception as e:
                    print(f"Warning: Could not rename {filename} ({e})")

            else:
                print(f"Warning: No mapping for {filename}, skipping.")

        else:
            print(f"Warning: Unexpected filename format: {filename}, skipping.")

# Example usage
reduction1_path = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/Reduction1/"
rename_fits_files(reduction1_path)

