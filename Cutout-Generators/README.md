# Cutout Generators

### build_cutouts_fast.py
Use this program to build image files from mosaic.

##### Terminal inputs:
module purge   
module load pytorch/2.11.0   
export PYTHONPATH=/global/common/software/desi/users/ioannis/SGAML/lib/python:/global/common/software/desi/users/ioannis/SGAML/lib/python3.12/site-packages   
export PATH=/global/common/software/desi/users/ioannis/SGAML/bin:$PATH   
export LD_LIBRARY_PATH=/global/common/software/desi/users/ioannis/SGAML/clib/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   
export SGA_DIR=/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025   
export SGA_DATA_DIR=/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data   
export SGA_HTML_DIR=/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025/html   

python build_ssl_cutouts.py --catalog /path/to/file.fits --region dr11-south --outdir /path/to/output --width 152 --pixscale 0.262 --chunksize 50000 --mp 32

##### History: 
python build_cutouts_fast.py --catalog /global/cfs/cdirs/desi/users/ioannis/SGA/2025/public/SGA2025-v1.0.fits --region dr11-south --outdir /pscratch/sd/q/qshimp/h5py-files --width 152 --pixscale 0.262 --chunksize 50000 --mp 128

python build_cutouts_fast.py --catalog /global/cfs/cdirs/desi/users/ioannis/SGA/2025/public/SGA2025-v1.0.fits --region dr11-north --outdir /pscratch/sd/q/qshimp/h5py-files --width 152 --pixscale 0.262 --chunksize 50000 --mp 128
### make_jpgs.py
Use this program to make image, model, and residual cutouts simultaneously.

##### Terminal inputs:
source /global/common/software/desi/desi_environment.sh main   
python make_jpgs.py --csv /path/to/file.csv --outdir /path/to/output

##### History:
For anchors 40" to 80"
python make_jpgs.py --csv /pscratch/sd/q/qshimp/Sorter/master_anchor_catalog.csv --outdir /pscratch/sd/q/qshimp/Cutouts/sga2025/Anchor_jpgs

For anchors 0" (25") to 40"
python make_jpgs.py --csv /global/cfs/cdirs/desicollab/users/qshimp/anchors/catalog_4400_0-40.csv --outdir /global/cfs/cdirs/desicollab/users/qshimp/cutouts/SGA-2025