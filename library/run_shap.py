from feature_selection_sp_test import sef_test
import sys
sys.path.append("..")
from cal_length_test_fs import scan_length_fs_shap,scan_length_fs_shap_topx

#sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_sentence.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_sentence_fs_shap_rmf_top100.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_sentence_fs_shap_filter_top100.txt')
sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_sentence.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_sentence_fs_shap_rmf.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_sentence_fs_shap_filter.txt')

#sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_pc_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_pc_token_fs_shap_rmf_top100.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_pc_token_fs_shap_filter_top100.txt')
sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_pc_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_pc_token_fs_shap_rmf.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_pc_token_fs_shap_filter.txt')

#sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_kmer_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_kmer_token_shap_rmf_top100.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_kmer_token_shap_filter_top100.txt')
sef_test('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_test_kmer_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_train_kmer_token_shap_rmf.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap/strains_test_kmer_token_shap_filter.txt')

#scan_length_fs_shap('../Ecoli_3fold/Fold3')
scan_length_fs_shap('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap')
#scan_length_fs_shap_topx('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/shap')
#scan_length_fs_shap('../Ecoli_3fold/Fold3')

