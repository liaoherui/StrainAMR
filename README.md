# StrainAMR
A learning-based tool to predict antimicrobial resistance and identify AMR-related genomic features from bacterial strain genomes

## Install (Linux or ubuntu only)

`git clone https://github.com/liaoherui/StrainAMR.git`<BR/>
`cd StrainAMR`<BR/>

`conda env create -f strainamr.yaml`<BR/>
`conda activate strainamr`<BR/>
`sh download_ps.sh`<BR/>
`python install_rebuild_ps.py`<BR/>

Test your installation：<BR/>

`python StrainAMR_build_train.py -h`<BR/>
`python StrainAMR_build_test.py -h`<BR/>
`python StrainAMR_model.py -h`<BR/>
