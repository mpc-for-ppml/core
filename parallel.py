import os
import subprocess

# Define the script names and their unique commands (Windows-style)
scripts = {
    "Core1.bat": "python secure_logreg.py -M3 -I0 data/demo_logreg_1/party1.csv -n zscore",
    "Core2.bat": "python secure_logreg.py -M3 -I1 data/demo_logreg_1/party2.csv -n zscore",
    "Core3.bat": "python secure_logreg.py -M3 -I2 data/demo_logreg_1/party3.csv -n zscore"
}

# Ensure the scripts directory exists
os.makedirs("scripts", exist_ok=True)

# Create and write each .bat script
for filename, command in scripts.items():
    with open(os.path.join("scripts", filename), 'w') as f:
        f.write(f'title {filename}\n{command}')

# Run them in separate terminal windows using full absolute path
for filename in scripts.keys():
    full_path = os.path.abspath(os.path.join("scripts", filename))
    subprocess.Popen(f'start "" cmd /k "{full_path}"', shell=True)
