import os
import shutil

# Define the path to the main folder containing the subfolders
main_folder = "/Users/piedeboer/Desktop/Thesis/code/signal-denoising/data/ecg-id"

# Loop through each subfolder
for person_folder in os.listdir(main_folder):
    # Construct the path to the subfolder
    person_folder_path = os.path.join(main_folder, person_folder)
    
    # Check if the path is a directory
    if os.path.isdir(person_folder_path):
        # Loop through each file in the subfolder
        for file_name in os.listdir(person_folder_path):
            # Construct the paths for the old and new files
            old_file_path = os.path.join(person_folder_path, file_name)
            new_file_path = os.path.join(main_folder, file_name)
            
            # Move the file to the main folder
            shutil.move(old_file_path, new_file_path)
        
        # Once all files are moved, remove the empty subfolder
        os.rmdir(person_folder_path)

print("All files moved to the main folder.")
