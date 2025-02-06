# Automatic-Translational-Corrction
This project is the part of the paper where translation values are generated through the matching network.
## Description
This project is the part of the paper where translation values are generated through the matching network. Specifically, we use a pre-trained matching model to generate the corresponding translation values for other views, using a reference view as the baseline.
## Requirements
```
# Create conda environment with torch 2.5.1 and CUDA 11.5
conda env create -f environment.yml
conda activate automatic_translate_correction
```
## Usage
1. Enter in the terminal:
   ```
   export PYTHONPATH=.../AutomaticTranslateCorrection:$PYTHONPATH
   ```
   Modify according to your own file path.
2. Configure your own config file, refer to ./project_config/ for details.
3. Enter in the terminal:
   ```
   python .../AutomaticTranslateCorrection/modules/translation_each_two_position.py
   ```
   Modify according to your own file path.
