# Using the Morphology Classifier on the SGA-2025
##### By Quillan Shimp

This is a guide to using the code in this folder. The primary notebook is `run_morphology_classifier.ipynb`. 

## Choose anchors
The classifier can handle anchors from any location by either pulling from existing embeddings or generating embeddings for them.  

### SGA-2025 anchors
Anchors that are cataloged in SGA-2025 all have existing embeddings. You need to use `umap_generator.ipynb` to add anchor information to the dataset. 

### Existing anchors
If you would like to use an existing set of anchors, set EXISTING_ANCHORS to true in `umap_generator.ipynb`. You can browse options in the Anchor-Generation folder. 