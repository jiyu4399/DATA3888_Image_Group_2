# DATA3888_Image_Group_2

In this project, teams are provided a Biotechnology Data Bundle, which contains a set of Mouse Brain cell images, alongside their cell identities (cluster ID). The goal is to develop and assess a computer vision tool to identify the identity of a cell (cluster ID), given an image of the cell’s morphology. Using machine learning or pre-trained deep learning models, teams will be tasked with communicating cell identity predictions, e.g. for unseen microscopy images. Each team will develop an interactive communication tool (e.g. Shiny app) to enable interactive assessment and/or prediction.

## Setup

As a team, we have decided to predominantly use Python for this project. As such, we have converted the given `R` code from the labs to Python so that we can reproduce every step of the process, including saving individual cell images using the whole `.tif` image file.

The first step is to set up the development environment to allow us to run all the code locally. Follow the steps below to do the same.

1. [Download](https://canvas.sydney.edu.au/courses/57772/files/35835364/download) the biotechnology data bundle
2. Download and install Python 3.10.0 [here](https://www.python.org/downloads/)
3. Clone this repository into your preferred location into the `scripts/` directory within the structure described below
4. Ensure Python is installed using `python3 -V` (the output should be `Python 3.10.0`)
5. From within the `scripts` directory, create a virtual environment using `python3 -m venv venv`
6. Deactivate any existing environments using `deactivate` or `conda deactivate`
7. Activate the created virtual environment using `source venv/bin/activate`
8. Install dependencies using `pip install -r requirements.txt`
9. Open the python notebook using VSCode or jupyter notebook and you should be ready to run the code! 

### Directory Structure
For the reproducible code we assume that the python notebook is saved within the following directory structure:

-   `Biotechnology/`
    -   `data_processed/`
        -   `clusters.csv`
        -   `cell_boundaries.csv.gz`
        -   `morphology_focus.tif`
    -   `data_raw/`
        -   `Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs.zip`
        -   `<unzipped files>`
    -   `scripts/`
        -    processing.ipynb _(from this repo)_
        -    DATA3888_Biotechnology_generateImages_2024.Rmd

For you to be able to fully re-run this code you will need to download the contents of  `data_raw/`  separately (see the Rmd file in the data bundle).
