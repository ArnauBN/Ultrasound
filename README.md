# Ultrasound
- Characterization of different materials (mainly epoxy resins) using ultrasounds.
- The `Arduino` folder contains the C++ code for the temperature measurements.
- `src` folder contains the custom libraries.
- Files starting with `ACQ` (Acquisition) in their name rely on serial commuincation with external devices to gather new data.
- Files starting with `SP` (Signal Processing) in their name load existing data from specific files.
- The environment.yml file is auto-generated with `conda export > environment.yml` and can be used to create a venv with the specified dependencies with `conda env create -f environment.yml`.
