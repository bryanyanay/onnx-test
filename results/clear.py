import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate over all files in the script's directory
for filename in os.listdir(script_dir):
    file_path = os.path.join(script_dir, filename)

    # Check if the path is a file and exclude the script file itself
    if os.path.isfile(file_path) and filename != 'clear.py':
        # Delete the file
        os.remove(file_path)
