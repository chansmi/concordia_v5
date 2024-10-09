import os

# Define the path to the folder
folder_path = 'concordia/factory/agent/agents'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Create the new filename by removing white spaces
    new_filename = filename.replace(' ', '')
    
    # Get the full path of the old and new filenames
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)
    
    # Rename the file only if the name has changed
    if old_file != new_file:
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')
    else:
        print(f'No change for: {filename}')