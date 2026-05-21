import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from bed_reader import open_bed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import xgboost as xgb



# 1. CONFIGURATION
RAW_PLINK_PREFIX = './data/raw_plink/nonGR_LONI_PPMI_MAY2023'
LINKLIST_PATH = './data/metadata/ppmi_244_linklist.csv'
LABEL_PATH = './data/metadata/gene_mapped.csv'

TEMP_DIR = './temp_plink'
FINAL_OUTPUT_DIR = './final_cv_folds'

PLINK_EXEC = 'plink'

N_SPLITS = 5
RANDOM_STATE = 42

GWAS_TOP_K = 100_000
FINAL_TOP_K = 100

LD_WINDOW = "50"
LD_STEP = "5"
LD_R2 = "0.5"

APPLY_FINAL_MINMAX_NORMALIZATION = True

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(FINAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 2. HELPER FUNCTIONS
def count_file_lines(filepath):
    """Efficiently count lines in text-based PLINK files such as .bim/.fam."""
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    with filepath.open("r") as f:
        return sum(1 for _ in f)


def run_plink(command_args):
    """Execute a PLINK command and fail fast if PLINK returns an error."""
    if shutil.which(PLINK_EXEC) is None and not Path(PLINK_EXEC).exists():
        raise FileNotFoundError(
            "PLINK executable not found. Please install PLINK or set PLINK_EXEC correctly."
        )

    base_cmd = [PLINK_EXEC] + [str(x) for x in command_args]
    print(f"\nRunning PLINK: {' '.join(base_cmd)}")
    result = subprocess.run(base_cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"PLINK ERROR:\n{result.stderr}")
        raise RuntimeError("PLINK command failed.")

    if result.stderr:
        print(result.stderr)


def find_plink_iid_column(link_df):
    """Find the column in the linklist that contains PPMI-style PLINK sample IDs."""
    candidate_cols = []
    for c in link_df.columns:
        values = link_df[c].astype(str)
        if values.str.contains("PPMI_", na=False).any():
            candidate_cols.append(c)

    if not candidate_cols:
        raise ValueError("Could not find a PPMI_* PLINK IID column in LINKLIST_PATH.")

    return candidate_cols[0]


def build_labelled_fam(qc_prefix):
    """
    Map PLINK FID/IID rows to PATNO and clinical labels.

    This function is used only to define valid subjects and CV splits.
    It does NOT perform GWAS, LD pruning, imputation, or feature selection.
    """
    fam_df = pd.read_csv(
        f"{qc_prefix}.fam",
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "Sex", "Pheno"],
    )
    link_df = pd.read_csv(LINKLIST_PATH)
    label_df = pd.read_csv(LABEL_PATH)

    fam_df["IID"] = fam_df["IID"].astype(str).str.strip()
    link_df["PATNO"] = link_df["PATNO"].astype(str).str.strip()
    label_df["PATNO"] = label_df["PATNO"].astype(str).str.strip()

    plink_col = find_plink_iid_column(link_df)
    link_df[plink_col] = link_df[plink_col].astype(str).str.strip()

    iid_to_patno = dict(zip(link_df[plink_col], link_df["PATNO"]))
    patno_to_label = dict(zip(label_df["PATNO"], label_df["Label_Int"]))

    fam_df["PATNO"] = fam_df["IID"].map(iid_to_patno)
    fam_df["Label_Int"] = fam_df["PATNO"].map(patno_to_label)

    valid_fam = fam_df.dropna(subset=["PATNO", "Label_Int"]).copy()
    valid_fam["Label_Int"] = valid_fam["Label_Int"].astype(int)

    return valid_fam.reset_index(drop=True)


def write_keep_file(fam_subset, filepath):
    """Write a PLINK --keep file with FID and IID only."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fam_subset[["FID", "IID"]].to_csv(
        filepath, sep="\t", index=False, header=False
    )


def write_train_pheno_file(fam_subset, filepath):
    """
    Write a PLINK phenotype file for the current training fold only.

    Labels are shifted by +10 so PLINK does not confuse class 0 with missing phenotypes.
    The phenotype is used only for within-training-fold association ranking.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    pheno_df = fam_subset[["FID", "IID", "Label_Int"]].copy()
    pheno_df["PLINK_Pheno"] = pheno_df["Label_Int"].astype(int) + 10
    pheno_df[["FID", "IID", "PLINK_Pheno"]].to_csv(
        filepath, sep="\t", index=False, header=False
    )


def read_gwas_and_get_top_snps(gwas_assoc_path, top_k):
    """
    Read PLINK --linear output and return the top-k SNPs sorted by p-value.

    PLINK .assoc.linear can contain non-ADD rows when covariates are present.
    If a TEST column exists, only ADD rows are used.
    """
    gwas_df = pd.read_csv(gwas_assoc_path, sep=r"\s+")

    if "TEST" in gwas_df.columns:
        gwas_df = gwas_df[gwas_df["TEST"].astype(str).str.upper() == "ADD"].copy()

    if "P" not in gwas_df.columns or "SNP" not in gwas_df.columns:
        raise ValueError(f"Unexpected GWAS output format: {gwas_assoc_path}")

    gwas_df["P"] = pd.to_numeric(gwas_df["P"], errors="coerce")
    gwas_df = (
        gwas_df.dropna(subset=["P", "SNP"])
        .sort_values(by="P", ascending=True)
        .drop_duplicates(subset=["SNP"])
    )

    if len(gwas_df) == 0:
        raise RuntimeError(f"No valid p-values found in {gwas_assoc_path}.")

    top_snps = gwas_df.head(min(top_k, len(gwas_df)))["SNP"].astype(str).tolist()
    return top_snps


def write_snp_list(snps, filepath):
    """Write one SNP ID per line."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        for snp in snps:
            f.write(f"{snp}\n")


def read_plink_matrix(plink_prefix):
    """Read a PLINK binary dataset into matrix, fam, and bim objects."""
    fam_df = pd.read_csv(
        f"{plink_prefix}.fam",
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "Sex", "Pheno"],
    )
    bim_df = pd.read_csv(
        f"{plink_prefix}.bim",
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP_ID", "cM", "BP", "A1", "A2"],
    )

    with open_bed(f"{plink_prefix}.bed") as bed:
        matrix = bed.read(dtype=np.float32)

    return matrix, fam_df, bim_df


def prepare_fold_plink_datasets(qc_prefix, train_fam, val_fam, fold_dir, fold):
    """
    For one CV fold:
    1) Run PLINK --linear only on the training fold.
    2) Retain top 100,000 training-ranked SNPs.
    3) Perform LD pruning only inside the training fold.
    4) Extract the training-derived pruned SNP list for both train and validation sets.
    """
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_keep = fold_dir / f"fold_{fold}_train.keep"
    val_keep = fold_dir / f"fold_{fold}_val.keep"
    train_pheno = fold_dir / f"fold_{fold}_train.pheno"

    write_keep_file(train_fam, train_keep)
    write_keep_file(val_fam, val_keep)
    write_train_pheno_file(train_fam, train_pheno)

    # Training-only GWAS association ranking
    gwas_out = fold_dir / "step2_train_only_gwas"
    run_plink([
        "--bfile", qc_prefix,
        "--keep", train_keep,
        "--pheno", train_pheno,
        "--linear",
        "--allow-no-sex",
        "--out", gwas_out,
    ])

    top_100k_snps = read_gwas_and_get_top_snps(
        f"{gwas_out}.assoc.linear", GWAS_TOP_K
    )
    top_100k_file = fold_dir / "fold_train_gwas_top100k.snplist"
    write_snp_list(top_100k_snps, top_100k_file)

    print(
        f"Fold {fold}: selected {len(top_100k_snps)} SNPs from training-only GWAS ranking."
    )

    # Create training-only top-100k dataset
    train_top100k_prefix = fold_dir / "step2_train_top100k"
    run_plink([
        "--bfile", qc_prefix,
        "--keep", train_keep,
        "--extract", top_100k_file,
        "--make-bed",
        "--allow-no-sex",
        "--out", train_top100k_prefix,
    ])

    # Training-only LD pruning
    ld_out = fold_dir / "step3_train_ld_prune"
    run_plink([
        "--bfile", train_top100k_prefix,
        "--indep-pairwise", LD_WINDOW, LD_STEP, LD_R2,
        "--allow-no-sex",
        "--out", ld_out,
    ])

    train_pruned_snps_file = fold_dir / "step3_train_ld_prune.prune.in"
    pruned_count = count_file_lines(train_pruned_snps_file)
    print(f"Fold {fold}: {pruned_count} SNPs retained after training-only LD pruning.")

    # Extract the SAME training-derived LD-pruned SNP columns for train and validation.
    train_pruned_prefix = fold_dir / "step3_train_pruned"
    val_pruned_prefix = fold_dir / "step3_val_pruned_training_snps"

    run_plink([
        "--bfile", qc_prefix,
        "--keep", train_keep,
        "--extract", train_pruned_snps_file,
        "--make-bed",
        "--allow-no-sex",
        "--out", train_pruned_prefix,
    ])

    run_plink([
        "--bfile", qc_prefix,
        "--keep", val_keep,
        "--extract", train_pruned_snps_file,
        "--make-bed",
        "--allow-no-sex",
        "--out", val_pruned_prefix,
    ])

    return str(train_pruned_prefix), str(val_pruned_prefix)

# 3. MAIN PIPELINE
def pipeline_master():
    print("STEP 1: GLOBAL UNSUPERVISED BASIC QC")
    print(
        "Note: supervised GWAS ranking, LD pruning, imputation, RF/XGBoost selection, "
        "and normalization are all performed inside each training fold below."
    )

    qc_temp = Path(TEMP_DIR) / "step1_qc_variants"
    qc_prefix = Path(TEMP_DIR) / "step1_qc_final"

    run_plink([
        "--bfile", RAW_PLINK_PREFIX,
        "--geno", "0.1",
        "--maf", "0.01",
        "--hwe", "1e-6",
        "--make-bed",
        "--allow-no-sex",
        "--out", qc_temp,
    ])

    print(
        f" -> STATS after variant QC: "
        f"{count_file_lines(f'{qc_temp}.bim')} SNPs | "
        f"{count_file_lines(f'{qc_temp}.fam')} individuals"
    )

    run_plink([
        "--bfile", qc_temp,
        "--mind", "0.1",
        "--make-bed",
        "--allow-no-sex",
        "--out", qc_prefix,
    ])

    print(
        f" -> STATS after individual QC: "
        f"{count_file_lines(f'{qc_prefix}.bim')} SNPs | "
        f"{count_file_lines(f'{qc_prefix}.fam')} individuals"
    )

    valid_fam = build_labelled_fam(str(qc_prefix))
    y_all = valid_fam["Label_Int"].values

    print("\nSTEP 2: SUBJECT-WISE 5-FOLD SPLIT")
    print(f" -> STATS: {len(valid_fam)} valid patients mapped to clinical labels.")
    print(" -> Class counts:")
    print(valid_fam["Label_Int"].value_counts().sort_index().to_string())

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    unique_labels = np.unique(y_all)
    label_map = {val: idx for idx, val in enumerate(unique_labels)}

    summary_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_fam, y_all), start=1):
        print(f"\n================ Processing Fold {fold}/{N_SPLITS} ================")

        train_fam = valid_fam.iloc[train_idx].copy()
        val_fam = valid_fam.iloc[val_idx].copy()

        fold_dir = Path(TEMP_DIR) / f"fold_{fold}"

        train_pruned_prefix, val_pruned_prefix = prepare_fold_plink_datasets(
            str(qc_prefix), train_fam, val_fam, fold_dir, fold
        )

        X_train_raw, train_matrix_fam, train_bim = read_plink_matrix(train_pruned_prefix)
        X_val_raw, val_matrix_fam, val_bim = read_plink_matrix(val_pruned_prefix)

        train_snp_names = train_bim["SNP_ID"].astype(str).values
        val_snp_names = val_bim["SNP_ID"].astype(str).values

        if list(train_snp_names) != list(val_snp_names):
            raise RuntimeError(
                f"Fold {fold}: train and validation SNP columns do not match. "
                "Validation must use only SNPs selected from the training fold."
            )

        # Re-map labels according to the actual .fam row order returned by PLINK.
        train_key_to_info = {
            (str(row.FID), str(row.IID)): (row.PATNO, int(row.Label_Int))
            for row in train_fam.itertuples(index=False)
        }
        val_key_to_info = {
            (str(row.FID), str(row.IID)): (row.PATNO, int(row.Label_Int))
            for row in val_fam.itertuples(index=False)
        }

        train_patnos, y_train = [], []
        for row in train_matrix_fam.itertuples(index=False):
            patno, label = train_key_to_info[(str(row.FID), str(row.IID))]
            train_patnos.append(patno)
            y_train.append(label)

        val_patnos, y_val = [], []
        for row in val_matrix_fam.itertuples(index=False):
            patno, label = val_key_to_info[(str(row.FID), str(row.IID))]
            val_patnos.append(patno)
            y_val.append(label)

        y_train = np.asarray(y_train, dtype=int)
        y_val = np.asarray(y_val, dtype=int)
        train_patnos = np.asarray(train_patnos).astype(str)
        val_patnos = np.asarray(val_patnos).astype(str)

        print(
            f"Fold {fold}: raw train matrix {X_train_raw.shape}, "
            f"raw validation matrix {X_val_raw.shape}"
        )

        # 1) Fit imputer on training only; transform validation using training statistics.
        imputer = SimpleImputer(strategy="most_frequent")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)

        # 2) RF feature importance fitted on training only.
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_model.fit(X_train_imp, y_train)
        rf_importances = minmax_scale(rf_model.feature_importances_)

        # 3) XGBoost feature importance fitted on training only.
        xgb_labels_train = np.array([label_map[l] for l in y_train])
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb_model.fit(X_train_imp, xgb_labels_train)
        xgb_importances = minmax_scale(xgb_model.feature_importances_)

        # 4) Training-only RF/XGBoost feature selection.
        combined_importances = (rf_importances + xgb_importances) / 2.0
        final_k = min(FINAL_TOP_K, X_train_imp.shape[1])
        top_indices = np.argsort(combined_importances)[::-1][:final_k]
        fold_snp_names = train_snp_names[top_indices]

        X_train_selected = X_train_imp[:, top_indices]
        X_val_selected = X_val_imp[:, top_indices]

        # 5) Optional normalization fitted on training only.
        if APPLY_FINAL_MINMAX_NORMALIZATION:
            scaler = MinMaxScaler()
            X_train_final = scaler.fit_transform(X_train_selected)
            X_val_final = scaler.transform(X_val_selected)
        else:
            X_train_final = X_train_selected
            X_val_final = X_val_selected

        # Save fold-specific selected SNP list.
        selected_snp_file = Path(FINAL_OUTPUT_DIR) / f"fold_{fold}_selected_top{final_k}_snps.txt"
        write_snp_list(fold_snp_names, selected_snp_file)

        # Save training set.
        df_train = pd.DataFrame(X_train_final, columns=fold_snp_names)
        df_train.insert(0, "PATNO", train_patnos)
        df_train.insert(1, "Label_Int", y_train)
        train_out = Path(FINAL_OUTPUT_DIR) / f"fold_{fold}_train_top{final_k}.csv"
        df_train.to_csv(train_out, index=False)

        # Save validation set with the exact same training-selected SNP columns.
        df_val = pd.DataFrame(X_val_final, columns=fold_snp_names)
        df_val.insert(0, "PATNO", val_patnos)
        df_val.insert(1, "Label_Int", y_val)
        val_out = Path(FINAL_OUTPUT_DIR) / f"fold_{fold}_val_top{final_k}.csv"
        df_val.to_csv(val_out, index=False)

        print(
            f"Fold {fold} saved successfully: "
            f"{train_out.name}, {val_out.name}; features={final_k}"
        )

        summary_rows.append({
            "fold": fold,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_after_train_gwas_top100k": count_file_lines(fold_dir / "fold_train_gwas_top100k.snplist"),
            "n_after_train_ld_prune": len(train_snp_names),
            "n_final_selected": final_k,
            "train_csv": str(train_out),
            "val_csv": str(val_out),
            "selected_snps": str(selected_snp_file),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(FINAL_OUTPUT_DIR) / "foldwise_preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n Fold-wise CV datasets saved to: {FINAL_OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    pipeline_master()
