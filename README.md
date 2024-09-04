# StrainAMR
A learning-based tool to predict antimicrobial resistance and identify multimodal AMR-related genomic features from bacterial strain genomes

## Install (Linux or ubuntu only)

`git clone https://github.com/liaoherui/StrainAMR.git`<BR/>
`cd StrainAMR`<BR/>
`unzip Test_genomes.zip`<BR/>
`unzip localDB.zip`<BR/>

`conda env create -f strainamr.yaml`<BR/>
`conda activate strainamr`<BR/>
`sh download_ps.sh`<BR/>
`python install_rebuild_ps.py`<BR/>

(Important) Add environment variable for running phenotypeseeker:

Open the bashrc file:
`vi ~/.bashrc`<BR/>

Add this line:
`export PATH=$PATH:<yout installation directory>/PhenotypeSeeker/.PSenv/bin`<BR/>
For example, if my installation dir is `/home/ray/StrainAMR`, then it should be<BR/>
`export PATH=$PATH:/home/ray/StrainAMR/PhenotypeSeeker/.PSenv/bin`<BR/>

Finally:
`source ~/.bashrc`<BR/>


Test your installation：<BR/>

`python StrainAMR_build_train.py -h`<BR/>
`python StrainAMR_build_test.py -h`<BR/>
`python StrainAMR_model_train.py -h`<BR/>
`python StrainAMR_model_pred.py -h`<BR/>

Test the tool with the demo data:<BR/>

`sh test_run.sh`

 Rerun the 3-fold cross validation experiment:<BR/>

 `sh batch_train_3fold_exp.sh`

