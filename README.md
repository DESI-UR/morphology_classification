# Ultimate Guide to Using the Morphology Classifier
##### By Quillan Shimp

###### Note from Julia
This repository contains code for machine-learning classification of galaxy morphologies developed during my time in the Cosmology Group at the University of Rochester. Originally developed to identify spiral galaxies for the DESI Tully-Fisher relation, the code has been extended to identify elliptical galaxies for the DESI Fundamental Plane analysis and used to perform morphological classification on the Siena Galaxy Atlas 2020. The code combines self-supervised learning and a k-nearest-neighbor classification to identify morphologies.


## Creating Target Dataset
The first step in classifying galaxies' morphologies is creating/formatting a dataset.
### Cutouts
The classifier runs on galaxy images with specific dimensions; your dataset must include these images, else you must generate them.
#### Small Datasets
You can use `make_cutouts.py` to generate a small batch of cutouts (I found it takes 10 minutes to generate 1000 cutouts) via the Legacy Survey Sky Browser. Make sure the LAYER is set to your desired data release. To run, follow these steps:
`cd morphology_classification`
`python make_cutouts.py --csv /file/path/to/targets.csv --outdir /folder/for/cutouts`
Replace /file/path/to/targets.csv with your file path and replace /folder/for/cutouts with your folder to store cutouts
##### Formatting FP_cutouts.py
In your data formatting file (See SGA_Cutouts.ipynb for an example), you should run a line that supplies the correct format to required columns:
`yourData = yourData.rename(columns={"yourRA": "Target_RA", "yourDec": "Target_DEC", "yourID": "TargetID"})`
The program uses IDs solely to name files, so datasets lacking IDs may assign arbitrary IDs to meet this requirement.

#### Large Datasets
For larger datasets, use [non-existent John path]




## Running the Morphology Classifier