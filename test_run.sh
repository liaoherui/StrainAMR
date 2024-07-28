python StrainAMR_build_train.py  -i Test_genomes/Sau/Ref_train -l Test_genomes/Sau/sau_label_train.txt -d levofloxacin -o Test_run/Sau_levofloxacin  &&\

python StrainAMR_build_test.py -i Test_genomes/Sau/Ref_val -l Test_genomes/Sau/sau_label_val.txt -d levofloxacin  -o Test_run/Sau_levofloxacin

