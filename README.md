# Ultimate Guide to Using the Morphology Classifier
##### By Quillan Shimp

###### Note from Julia
This repository contains code for machine-learning classification of galaxy morphologies developed during my time in the Cosmology Group at the University of Rochester. Originally developed to identify spiral galaxies for the DESI Tully-Fisher relation, the code has been extended to identify elliptical galaxies for the DESI Fundamental Plane analysis and used to perform morphological classification on the Siena Galaxy Atlas 2020. The code combines self-supervised learning and a k-nearest-neighbor classification to identify morphologies.


## Creating Target Dataset
The first step in classifying galaxies' morphologies is creating/formatting a dataset.
### Generating Cutouts
The classifier runs on galaxy images with specific dimensions; your dataset must include these images, else you must generate them.
#### Small Datasets
You can use `make_cutouts.py` to generate a small batch of cutouts (I found it takes 10 minutes to generate 1000 cutouts) via the Legacy Survey Sky Browser. Make sure the LAYER is set to your desired data release. To run, follow these steps:
`cd morphology_classification`
`source /global/common/software/desi/desi_environment.sh main`
`python make_cutouts.py --csv /file/path/to/targets.csv --outdir /folder/for/cutouts`
Replace /file/path/to/targets.csv with your file path and replace /folder/for/cutouts with your folder to store cutouts.
NOTE: For some reason that I haven't bothered troubleshooting, cutouts are generated vertically flipped.
##### Formatting FP_cutouts.py
In your data formatting file (See /SGA/SGA_Cutouts.ipynb for an example), you should run a line that supplies the correct format to required columns:
`yourData = yourData.rename(columns={"yourRA": "Target_RA", "yourDec": "Target_DEC", "yourID": "TargetID"})`
The program uses IDs solely to name files, so datasets lacking IDs may assign arbitrary IDs to meet this requirement.

#### Large Datasets
For larger datasets, use [non-existent John path\]

### Appending Cutout Paths and Anchors
As of now, you should use /SGA/SGA_Generator.ipynb to append cutout paths to your target dataset. Currently, you must read your dataset, give the program the directory to your cutouts, plug in your specific names into the initializer, change the length to your length, and change the write location to a file of your choosing. 

The path finder is configured to the current version of make_cutouts.py, you will have to change the path style if you are using a different cutout generator. 

I'm considering turning it into a function so you merely need to input the directory, file name, and specify column names instead of adjusting the code yourself. Also when I get John's program I will make an option to specify whether it was generated with the web version or John's version. 

You must add a main type column to your target dataset. Give it the value 30. Now append this file 
`/pscratch/sd/q/qshimp/SGA2020-data/Anchors/VI_4000_sga152x152.fits`
Now you can save this file.

My code currently does this very clunkily going back and forth between SGA generator and SGA cutouts. I'm going to move it all to generator and make it a single function for less user input.

WARNING: If your ID system is not the same, you will need to check for duplicates and add a small id tag that can distinguish duplicates without creating more duplicates. I had just one duplicate pair between SGA id and SGA 2025 beta's ref id, and adding a 0 to the end of the ref id worked just fine.

### Creating the h5py File
All formatting should be complete, and you should be able to simply change the input file that gal_dataset reads on /h5py-code/h5py-Generator.ipynb. 
If you run into issues, try changing the formatting to match the h5py file. If your input dataset changes the path datatype, you will need to manually edit that in SGA generator before running the h5py generator. I haven't figured out how to retain the byte datatype properly in a CSV file yet. 
## Running the Morphology Classifier
Before running your morphology classifier for the first time, ensure that it has all the files and directory information that it needs. The cleanest way to do this is to have a copy of ssl-legacysurvey outside of morphology_classification and to move run_morphology_classifier.ipynb into this folder. You also will need to move resnet50.ckpt and create a folder called diskdata.