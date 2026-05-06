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


# HELPER FUNCTIONS

def count_file_lines(filepath):
    """Efficiently counts the number of lines in a file (used for .bim and .fam stats)."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def run_plink(command_args):
    """Executes PLINK commands via Python subprocess."""
    if shutil.which(PLINK_EXEC) is None and not os.path.exists(PLINK_EXEC):
        raise FileNotFoundError(f"PLINK executable not found. Please install it on the GCP instance.")

    base_cmd = [PLINK_EXEC] + command_args
    print(f"Running PLINK: {' '.join(base_cmd)}")
    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PLINK ERROR:\n{result.stderr}")
        raise RuntimeError("PLINK command failed.")

def build_phenotype_file(qc_prefix):
    """Maps labels for PLINK phenotyping."""
    fam_df = pd.read_csv(f"{qc_prefix}.fam", sep=r'\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
    link_df = pd.read_csv(LINKLIST_PATH)
    label_df = pd.read_csv(LABEL_PATH)

    link_df['PATNO'] = link_df['PATNO'].astype(str).str.strip()
    label_df['PATNO'] = label_df['PATNO'].astype(str).str.strip()
    fam_df['IID'] = fam_df['IID'].astype(str).str.strip()

    plink_col = [c for c in link_df.columns if link_df[c].astype(str).str.contains('PPMI_').any()][0]
    patno_to_label = dict(zip(label_df['PATNO'], label_df['Label_Int']))
    link_df['Label_Int'] = link_df['PATNO'].map(patno_to_label)

    plink_to_label = dict(zip(link_df[plink_col], link_df['Label_Int']))
    fam_df['New_Pheno'] = fam_df['IID'].map(plink_to_label)
    fam_df['New_Pheno'] = fam_df['New_Pheno'].fillna(-9).astype(int)

    # +10 Shift for PLINK math safety
    valid_mask = fam_df['New_Pheno'] != -9
    fam_df.loc[valid_mask, 'New_Pheno'] += 10

    pheno_file = os.path.join(TEMP_DIR, "custom_pheno.txt")
    fam_df[['FID', 'IID', 'New_Pheno']].to_csv(pheno_file, sep='\t', index=False, header=False)
    return pheno_file


# MAIN PIPELINE
def pipeline_master():
    print(" STEP 1: QC ")
    QC_TEMP = os.path.join(TEMP_DIR, "step1_qc_variants")

    run_plink(["--bfile", RAW_PLINK_PREFIX, "--geno", "0.1", "--maf", "0.01", "--hwe", "1e-6", "--make-bed", "--allow-no-sex", "--out", QC_TEMP])

    print(f" -> STATS (Post Variant QC): {count_file_lines(f'{QC_TEMP}.bim')} SNPs | {count_file_lines(f'{QC_TEMP}.fam')} Individuals")

    QC_PREFIX = os.path.join(TEMP_DIR, "step1_qc_final")
    run_plink(["--bfile", QC_TEMP, "--mind", "0.1", "--make-bed", "--allow-no-sex", "--out", QC_PREFIX])

    print(f" -> STATS (Post Individual QC): {count_file_lines(f'{QC_PREFIX}.bim')} SNPs | {count_file_lines(f'{QC_PREFIX}.fam')} Individuals")

    print("\nSTEP 2: GWAS & EXTRACTION ")
    pheno_file = build_phenotype_file(QC_PREFIX)
    run_plink(["--bfile", QC_PREFIX, "--pheno", pheno_file, "--linear", "--allow-no-sex", "--out", os.path.join(TEMP_DIR, "step2_gwas")])

    gwas_df = pd.read_csv(os.path.join(TEMP_DIR, "step2_gwas.assoc.linear"), sep=r'\s+')
    gwas_df = gwas_df.dropna(subset=['P']).sort_values(by='P')


    top_100k_snps = gwas_df.head(100000)['SNP'].tolist()

    extract_file = os.path.join(TEMP_DIR, "top_snps_extract.txt")
    with open(extract_file, 'w') as f:
        for snp in top_100k_snps: f.write(f"{snp}\n")

    run_plink(["--bfile", QC_PREFIX, "--extract", extract_file, "--make-bed", "--out", os.path.join(TEMP_DIR, "step2_retained")])

    retained_snps = count_file_lines(os.path.join(TEMP_DIR, "step2_retained.bim"))
    print(f" -> STATS: Extracted {retained_snps} top GWAS candidates into new dataset.")

    print("\nSTEP 3: LD PRUNING ")

    run_plink(["--bfile", os.path.join(TEMP_DIR, "step2_retained"), "--indep-pairwise", "50", "5", "0.5", "--out", os.path.join(TEMP_DIR, "ld_prune")])


    run_plink(["--bfile", os.path.join(TEMP_DIR, "step2_retained"), "--extract", os.path.join(TEMP_DIR, "ld_prune.prune.in"), "--make-bed", "--out", os.path.join(TEMP_DIR, "step3_pruned")])

    final_pruned_count = count_file_lines(os.path.join(TEMP_DIR, "step3_pruned.bim"))
    print(f" -> STATS: {final_pruned_count} total SNPs retained after natural LD Pruning.")

    # Mapping Labels...
    reduced_fam = pd.read_csv(os.path.join(TEMP_DIR, "step3_pruned.fam"), sep=r'\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Pheno'])
    reduced_bim = pd.read_csv(os.path.join(TEMP_DIR, "step3_pruned.bim"), sep=r'\s+', header=None, names=['CHR', 'SNP_ID', 'cM', 'BP', 'A1', 'A2'])

    link_df_reverse = pd.read_csv(LINKLIST_PATH)
    plink_col = [c for c in link_df_reverse.columns if link_df_reverse[c].astype(str).str.contains('PPMI_').any()][0]
    iid_to_patno = dict(zip(link_df_reverse[plink_col].astype(str).str.strip(), link_df_reverse['PATNO'].astype(str).str.strip()))
    raw_patnos = [iid_to_patno.get(str(iid), str(iid)) for iid in reduced_fam['IID'].values]

    label_df_real = pd.read_csv(LABEL_PATH)
    label_df_real['PATNO'] = label_df_real['PATNO'].astype(str).str.strip()
    patno_to_real_label = dict(zip(label_df_real['PATNO'], label_df_real['Label_Int']))

    real_labels = np.array([patno_to_real_label.get(str(p), np.nan) for p in raw_patnos])
    valid_patient_mask = ~np.isnan(real_labels)
    final_real_labels = real_labels[valid_patient_mask].astype(int)
    final_patnos = np.array(raw_patnos)[valid_patient_mask]

    print("\n STEP 4: 5-FOLD CV IMPUTATION & PURE DATA-DRIVEN SELECTION")
    print(f" -> STATS: {len(final_patnos)} valid patients successfully mapped to clinical labels.")


    with open_bed(os.path.join(TEMP_DIR, "step3_pruned.bed")) as bed:
        raw_matrix = bed.read(dtype=np.float32)[valid_patient_mask, :]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    actual_snps_in_file = reduced_bim['SNP_ID'].values

    unique_labels = np.unique(final_real_labels)
    label_map = {val: idx for idx, val in enumerate(unique_labels)}

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_matrix, final_real_labels)):
        print(f"\n Processing Fold {fold + 1}/5 ")
        X_train, X_val = raw_matrix[train_idx], raw_matrix[val_idx]
        y_train, y_val = final_real_labels[train_idx], final_real_labels[val_idx]
        patnos_train, patnos_val = final_patnos[train_idx], final_patnos[val_idx]

        # 1. Fit Imputer only on Training Data
        imputer = SimpleImputer(strategy='most_frequent')
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        # 2. Fit Random Forest only on Training Data
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_imp, y_train)
        rf_importances = minmax_scale(rf_model.feature_importances_)

        # 3. Fit XGBoost only on Training Data
        xgb_labels_train = np.array([label_map[l] for l in y_train])
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_imp, xgb_labels_train)
        xgb_importances = minmax_scale(xgb_model.feature_importances_)

        # 4 Data-Driven Selection 
        combined_importances = (rf_importances + xgb_importances) / 2.0
        top_100_indices = np.argsort(combined_importances)[::-1][:100]

        # 5. Extract and save Fold-Specific datasets
        X_train_final = X_train_imp[:, top_100_indices]
        X_val_final = X_val_imp[:, top_100_indices]
        fold_snp_names = actual_snps_in_file[top_100_indices]

        # Save Training Set
        df_train = pd.DataFrame(X_train_final, columns=fold_snp_names)
        df_train.insert(0, 'PATNO', patnos_train)
        df_train.insert(1, 'Label_Int', y_train)
        df_train.to_csv(os.path.join(FINAL_OUTPUT_DIR, f'fold_{fold+1}_train_top100.csv'), index=False)

        # Save Validation Set
        df_val = pd.DataFrame(X_val_final, columns=fold_snp_names)
        df_val.insert(0, 'PATNO', patnos_val)
        df_val.insert(1, 'Label_Int', y_val)
        df_val.to_csv(os.path.join(FINAL_OUTPUT_DIR, f'fold_{fold+1}_val_top100.csv'), index=False)

        print(f"Fold {fold + 1} saved successfully. Features: {X_train_final.shape[1]}")

    print(f"\nSUCCESS! 5-Fold Cross-Validated datasets saved to: {FINAL_OUTPUT_DIR}")

if __name__ == '__main__':
    pipeline_master()