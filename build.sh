# Step 1. Training data Feature Extraction
#strainamr build-train -i ../../data/genomes-train -o Kpn-imipenem-Features/db -l ../../data/train_label.txt -d imipenem -t 32 --no-kmer-weights --cluster-tool cd-hit --amr-organism Klebsiella_pneumoniae

# Step 2. Val data Feature Extraction
#strainamr build-test   -i ../../data/genomes-val   -l ../../data/val_label.txt   -d imipenem   -o Kpn-imipenem-Features/val_features  --db Kpn-imipenem-Features/db  -t 32

# Step 3. Test data Feature Extraction
#strainamr build-test   -i ../../data/genomes-test   -l ../../data/test_label.txt   -d imipenem   -o Kpn-imipenem-Features/test_features  --db Kpn-imipenem-Features/db  -t 32

# Step 4. Database Slim (Delete Tem Files -> Save Disk Space)
#strainamr slim-db Kpn-imipenem-Features/db  Kpn-imipenem-Features/val_features Kpn-imipenem-Features/test_features  --apply

# Step 5. Training
#CUDA_VISIBLE_DEVICES=0 strainamr train -i Kpn-imipenem-Features/db --val-dir Kpn-imipenem-Features/val_features -o res-Kpn-imipenem \
#  -s 1 --patience 20 --epochs 100 2>&1 | tee logs/log-train-Kpn-imipenem.txt

# Step 6. Prediction
#CUDA_VISIBLE_DEVICES=0 strainamr predict -i Kpn-imipenem-Features/test_features -m res-Kpn-imipenem -o pred-Kpn-imipenem --db Kpn-imipenem-Features/db 2>&1 | tee logs/log-predict-Kpn-imipenem_re.txt

# No need to run the command line below
#strainamr mechanism-eval  --db Kpn-imipenem-Features/db --features Kpn-imipenem-Features/test_features  --pred pred-Kpn-imipenem --model res-Kpn-imipenem --val-features Kpn-imipenem-Features/val_features -o pred-Kpn-imipenem 
