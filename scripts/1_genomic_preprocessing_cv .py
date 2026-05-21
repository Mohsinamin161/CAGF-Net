import pandas as pd
import numpy as np
import subprocess
import os
import shutil
from bed_reader import open_bed
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# 1. CONFIGURATION
RAW_PLINK_PREFIX = './data/raw_plink/nonGR_LONI_PPMI_MAY2023'
LINKLIST_PATH = './data/metadata/ppmi_244_linklist.csv'
LABEL_PATH = './data/metadata/gene_mapped.csv'

TEMP_DIR = './temp_plink'
FINAL_OUTPUT_DIR = './final_cv_folds'
PLINK_EXEC = 'plink'

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


# 2. HELPER FUNCTIONS
def count_file_lines(filepath):
    if not os.path.exists(filepath): return 0
    with open(filepath, 'r') as f: return sum(1 for _ in f)

def run_plink(command_args):
    if shutil.which(PLINK_EXEC) is None and not os.path.exists(PLINK_EXEC):
        raise FileNotFoundError("PLINK executable not found. Please install it.")
    base_cmd = [PLINK_EXEC] + command_args
    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PLINK ERROR:\n{result.stderr}")
        raise RuntimeError("PLINK command failed.")

# 3. MAIN PIPELINE
def pipeline_master():
    print("--- STEP 1: GLOBAL UNSUPERVISED QC ---")
    QC_TEMP = os.path.join(TEMP_DIR, "step1_qc_variants")
    run_plink(["--bfile", RAW_PLINK_PREFIX, "--geno", "0.1", "--maf", "0.01", "--hwe", "1e-6", "--make-bed", "--allow-no-sex", "--out", QC_TEMP])
    print(f" -> STATS (Post Variant QC): {count_file_lines(f'{QC_TEMP}.bim')} SNPs")

    QC_PREFIX = os.path.join(TEMP_DIR, "step1_qc_final")
    run_plink(["--bfile", QC_TEMP, "--mind", "0.1", "--make-bed", "--allow-no-sex", "--out", QC_PREFIX])
    print(f" -> STATS (Post Individual QC): {count_file_lines(f'{QC_PREFIX}.fam')} Individuals")

    print("\n--- STEP 2: PREPARING CLINICAL MAPPING ---")
    # Read the PLINK subjects
    fam_df = pd.read_csv(f"{QC_PREFIX}.fam", sep=r'\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
    fam_df['IID'] = fam_df['IID'].astype(str).str.strip()

    # Read Metadata
    link_df = pd.read_csv(LINKLIST_PATH)
    label_df = pd.read_csv(LABEL_PATH)
    
    plink_col = [c for c in link_df.columns if link_df[c].astype(str).str.contains('PPMI_').any()][0]
    iid_to_patno = dict(zip(link_df[plink_col].astype(str).str.strip(), link_df['PATNO'].astype(str).str.strip()))
    patno_to_label = dict(zip(label_df['PATNO'].astype(str).str.strip(), label_df['Label_Int']))

    # Filter for valid subjects that have a label
    valid_iids, valid_labels = [], []
    iid_to_final_label = {}

    for iid in fam_df['IID']:
        patno = iid_to_patno.get(iid)
        if patno and patno in patno_to_label:
            lbl = int(patno_to_label[patno])
            valid_iids.append(iid)
            valid_labels.append(lbl)
            iid_to_final_label[iid] = lbl

    valid_iids = np.array(valid_iids)
    valid_labels = np.array(valid_labels)
    print(f" -> Successfully mapped {len(valid_iids)} subjects to clinical labels.")

    print("\n--- STEP 3:FOLD-WISE FEATURE SELECTION ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_iids, valid_labels)):
        print(f"\n==========================================")
        print(f" PROCESSING FOLD {fold + 1}/5")
        print(f"==========================================")
        
        train_iids = set(valid_iids[train_idx])
        val_iids = set(valid_iids[val_idx])

        # 3.1 MASKING PHENOTYPES FOR PLINK
        fold_pheno_path = os.path.join(TEMP_DIR, f"pheno_fold_{fold+1}.txt")
        fold_fam = fam_df.copy()
        
        def assign_pheno(iid):
            if iid in train_iids:
                return iid_to_final_label[iid] + 10 # +10 shift for PLINK linear regression safety
            return -9 # PLINK ignores -9 during association tests
            
        fold_fam['New_Pheno'] = fold_fam['IID'].apply(assign_pheno)
        fold_fam[['FID', 'IID', 'New_Pheno']].to_csv(fold_pheno_path, sep='\t', index=False, header=False)

        # 3.2  GWAS ON TRAINING DATA
        print(" -> Running PLINK --linear GWAS on Training subset...")
        gwas_out = os.path.join(TEMP_DIR, f"fold_{fold+1}_gwas")
        run_plink(["--bfile", QC_PREFIX, "--pheno", fold_pheno_path, "--linear", "--allow-no-sex", "--out", gwas_out])

        gwas_df = pd.read_csv(f"{gwas_out}.assoc.linear", sep=r'\s+')
        gwas_df = gwas_df.dropna(subset=['P']).sort_values(by='P')
        top_100k_snps = gwas_df.head(100000)['SNP'].tolist()

        extract_file = os.path.join(TEMP_DIR, f"fold_{fold+1}_extract.txt")
        with open(extract_file, 'w') as f:
            for snp in top_100k_snps: f.write(f"{snp}\n")

        # Extract top 100k for all subjects (so we have validation features later)
        retained_out = os.path.join(TEMP_DIR, f"fold_{fold+1}_retained")
        run_plink(["--bfile", QC_PREFIX, "--extract", extract_file, "--make-bed", "--out", retained_out])

        # 3.3 LD PRUNING
        print(" -> Running LD Pruning on fold-specific GWAS candidates...")
        ld_out = os.path.join(TEMP_DIR, f"fold_{fold+1}_ld")
        run_plink(["--bfile", retained_out, "--indep-pairwise", "50", "5", "0.5", "--out", ld_out])
        
        pruned_out = os.path.join(TEMP_DIR, f"fold_{fold+1}_pruned")
        run_plink(["--bfile", retained_out, "--extract", f"{ld_out}.prune.in", "--make-bed", "--out", pruned_out])
        print(f" -> STATS: {count_file_lines(f'{pruned_out}.bim')} SNPs retained after LD Pruning.")

        # 3.4 LOAD FOLD-SPECIFIC MATRICES INTO PYTHON
        print(" -> Loading Fold Data into Python...")
        pruned_fam = pd.read_csv(f"{pruned_out}.fam", sep=r'\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
        pruned_bim = pd.read_csv(f"{pruned_out}.bim", sep=r'\s+', header=None, names=['CHR', 'SNP_ID', 'cM', 'BP', 'A1', 'A2'])
        
        pruned_iids_array = pruned_fam['IID'].astype(str).str.strip().values
        actual_snps_in_file = pruned_bim['SNP_ID'].values

        with open_bed(f"{pruned_out}.bed") as bed:
            raw_matrix = bed.read(dtype=np.float32)

        # 3.5 STRICT SEPARATION OF TRAIN AND VAL TENSORS
        train_mask = np.array([iid in train_iids for iid in pruned_iids_array])
        val_mask = np.array([iid in val_iids for iid in pruned_iids_array])

        X_train, X_val = raw_matrix[train_mask], raw_matrix[val_mask]
        y_train = np.array([iid_to_final_label[iid] for iid in pruned_iids_array[train_mask]])
        y_val = np.array([iid_to_final_label[iid] for iid in pruned_iids_array[val_mask]])
        patnos_train = np.array([iid_to_patno[iid] for iid in pruned_iids_array[train_mask]])
        patnos_val = np.array([iid_to_patno[iid] for iid in pruned_iids_array[val_mask]])

        # 3.6 FOLD-WISE IMPUTATION
        imputer = SimpleImputer(strategy='most_frequent')
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        # 3.7 DATA-DRIVEN ENSEMBLE SELECTION
        print(" -> Running RF/XGBoost Feature Selection on Training Subset...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_imp, y_train)
        rf_importances = minmax_scale(rf_model.feature_importances_)

        unique_labels = np.unique(valid_labels)
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        xgb_labels_train = np.array([label_map[l] for l in y_train])
        
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_imp, xgb_labels_train)
        xgb_importances = minmax_scale(xgb_model.feature_importances_)

        combined_importances = (rf_importances + xgb_importances) / 2.0
        top_100_indices = np.argsort(combined_importances)[::-1][:100]

        # 3.8 SAVE FINAL FOLD DATASETS
        X_train_final = X_train_imp[:, top_100_indices]
        X_val_final = X_val_imp[:, top_100_indices]
        fold_snp_names = actual_snps_in_file[top_100_indices]

        df_train = pd.DataFrame(X_train_final, columns=fold_snp_names)
        df_train.insert(0, 'PATNO', patnos_train)
        df_train.insert(1, 'Label_Int', y_train)
        df_train.to_csv(os.path.join(FINAL_OUTPUT_DIR, f'fold_{fold+1}_train_top100.csv'), index=False)

        df_val = pd.DataFrame(X_val_final, columns=fold_snp_names)
        df_val.insert(0, 'PATNO', patnos_val)
        df_val.insert(1, 'Label_Int', y_val)
        df_val.to_csv(os.path.join(FINAL_OUTPUT_DIR, f'fold_{fold+1}_val_top100.csv'), index=False)

        print(f" -> Fold {fold + 1} completed successfully.")

    print(f"\n isolated 5-Fold datasets saved to: {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    pipeline_master()
