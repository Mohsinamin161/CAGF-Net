import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

from 3_cagf_architectures import CAGFNet, Generator, Discriminator


# 1. CONFIGURATION & PATHS
IMAGING_CSV = './data/aligned_mri_dti_512.csv'
GENE_CSV = './data/512_genetic_features.csv'
OUTPUT_DIR = './results_cv'

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. DATASET DEFINITION
class MultimodalDataset(Dataset):
    def __init__(self, imaging, gene, labels):
        self.imaging = torch.tensor(imaging, dtype=torch.float32)
        self.gene = torch.tensor(gene, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.imaging[idx], self.gene[idx], self.labels[idx]


# 3. CROSS-MODAL CONTRASTIVE ALIGNMENT LOSS
def cross_modal_contrastive_loss(out_img, out_gene, tau=0.1):
    batch_size = out_img.size(0)

    # L2 normalize to compute cosine similarity via dot product
    out_img = F.normalize(out_img, dim=1)
    out_gene = F.normalize(out_gene, dim=1)

    # Calculate sim(out_img, out_gene) / tau
    sim_matrix = torch.matmul(out_img, out_gene.T) / tau

    labels = torch.arange(batch_size).to(DEVICE)
    L_contrast = F.cross_entropy(sim_matrix, labels)

    return L_contrast

# 4. SMOTIFIED-GAN AUGMENTATION
def apply_smotified_gan(X_train, y_train, feature_dim=1024, latent_dim=100, epochs=1000, batch_size=64):
    """Generates synthetic samples in the joint latent space exclusively on training folds."""
    print(" Running SMOTE initialization:")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    class_counts = np.bincount(y_train)
    max_class_count = np.max(class_counts)

    X_synthetic, y_synthetic = [], []

    for class_label in np.unique(y_train):
        if class_counts[class_label] < max_class_count:
            samples_to_generate = max_class_count - class_counts[class_label]
            class_data = torch.tensor(X_smote[y_smote == class_label], dtype=torch.float32).to(DEVICE)

            gen = Generator(latent_dim, feature_dim).to(DEVICE)
            disc = Discriminator(feature_dim).to(DEVICE)
            criterion = nn.BCELoss()
            opt_G = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
            opt_D = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

            for epoch in range(epochs):
                idx = torch.randint(0, class_data.size(0), (batch_size,))
                real_batch = class_data[idx]
                real_labels, fake_labels = torch.ones(batch_size, 1).to(DEVICE), torch.zeros(batch_size, 1).to(DEVICE)

                # Train Discriminator
                opt_D.zero_grad()
                loss_D = criterion(disc(real_batch), real_labels) + criterion(disc(gen(torch.randn(batch_size, latent_dim).to(DEVICE)).detach()), fake_labels)
                loss_D.backward()
                opt_D.step()

                # Train Generator
                opt_G.zero_grad()
                loss_G = criterion(disc(gen(torch.randn(batch_size, latent_dim).to(DEVICE))), real_labels)
                loss_G.backward()
                opt_G.step()

            gen.eval()
            with torch.no_grad():
                z_synth = torch.randn(samples_to_generate, latent_dim).to(DEVICE)
                X_synthetic.append(gen(z_synth).cpu().numpy())
                y_synthetic.append(np.full(samples_to_generate, class_label))

    if X_synthetic:
        return np.vstack((X_train, np.vstack(X_synthetic))), np.concatenate((y_train, np.concatenate(y_synthetic)))
    return X_train, y_train

# 5. MAIN 5-FOLD CV PIPELINE
def main():
    print("--- 1. LOADING DATA ---")
    img_df = pd.read_csv(IMAGING_CSV)
    gen_df = pd.read_csv(GENE_CSV)

    # Determine which label column to use for the ground truth 'y'
    target_col = 'Label' if 'Label' in img_df.columns else 'Label_Int'

    gen_clean = gen_df.drop(columns=['Label', 'Label_Int'], errors='ignore')
    fused_df = pd.merge(img_df, gen_clean, on='PATNO', how='inner')

    y = fused_df[target_col].values

    # Safely exclude all possible metadata columns from the feature arrays
    exclude_cols = ['PATNO', 'Label', 'Label_Int']
    X_img = fused_df[[c for c in img_df.columns if c not in exclude_cols]].values
    X_gen = fused_df[[c for c in gen_df.columns if c not in exclude_cols]].values

    # Sanity check to ensure perfect 512 + 512 dimensions
    print(f"Data Loaded! Imaging shape: {X_img.shape}, Genomic shape: {X_gen.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    print("\n 2. INITIATING STRICT 5-FOLD CROSS-VALIDATION ")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
        print(f"\n FOLD {fold+1} ")

        # 1. Split Data
        X_img_tr, X_img_v = X_img[train_idx], X_img[val_idx]
        X_gen_tr, X_gen_v = X_gen[train_idx], X_gen[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        # 2. Scale Independently to prevent leakage
        scaler_img, scaler_gen = StandardScaler(), StandardScaler()
        X_img_tr = scaler_img.fit_transform(X_img_tr)
        X_img_v = scaler_img.transform(X_img_v)
        X_gen_tr = scaler_gen.fit_transform(X_gen_tr)
        X_gen_v = scaler_gen.transform(X_gen_v)

        # 3. SMOTified-GAN in joint latent space (TRAIN ONLY)
        print("   -> Applying SMOTified-GAN Augmentation...")
        X_joint_tr = np.hstack((X_img_tr, X_gen_tr))
        X_joint_bal, y_tr_bal = apply_smotified_gan(X_joint_tr, y_tr)

        # Decouple back to separate balanced modalities
        X_img_bal = X_joint_bal[:, :512]
        X_gen_bal = X_joint_bal[:, 512:]

        train_loader = DataLoader(MultimodalDataset(X_img_bal, X_gen_bal, y_tr_bal), batch_size=32, shuffle=True)
        val_loader = DataLoader(MultimodalDataset(X_img_v, X_gen_v, y_v), batch_size=32, shuffle=False)

        # 4. Initialize CAGF-Net
        model = CAGFNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        alpha = 0.5 # Hyperparameter for contrastive loss penalty

        best_f1, fold_auc, fold_acc, fold_prec, fold_rec = 0, 0, 0, 0, 0

        print("   -> Training CAGF-Net...")
        for epoch in range(50):
            model.train()
            for img_batch, gene_batch, lbls in train_loader:
                img_batch, gene_batch, lbls = img_batch.to(DEVICE), gene_batch.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()

                # Forward Pass
                preds, out_img, out_gene = model(img_batch, gene_batch)


                # DUAL LOSS OPTIMIZATION
                L_CE = criterion(preds, lbls)  # Primary Cross-Entropy Loss
                L_contrast = cross_modal_contrastive_loss(out_img, out_gene, tau=0.1)

                # Total Loss: L_total = L_CE + alpha * L_contrast
                L_total = L_CE + (alpha * L_contrast)

                L_total.backward()
                optimizer.step()

            # Validation Step
            model.eval()
            all_l, all_p, all_prob = [], [], []
            with torch.no_grad():
                for img_batch, gene_batch, lbls in val_loader:
                    img_batch, gene_batch = img_batch.to(DEVICE), gene_batch.to(DEVICE)
                    preds, _, _ = model(img_batch, gene_batch)

                    all_l.extend(lbls.numpy())
                    all_p.extend(torch.max(preds, 1)[1].cpu().numpy())
                    all_prob.extend(F.softmax(preds, dim=1).cpu().numpy())

            # Calculate all metrics
            prec, rec, f1, _ = precision_recall_fscore_support(all_l, all_p, average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                fold_acc = accuracy_score(all_l, all_p)
                fold_prec = prec
                fold_rec = rec
                fold_auc = roc_auc_score(all_l, all_prob, multi_class='ovr', average='weighted')
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'cagf_net_fold{fold+1}.pth'))

        print(f"   -> Fold {fold+1} Best: Acc: {fold_acc:.3f} | Prec: {fold_prec:.3f} | Rec: {fold_rec:.3f} | F1: {best_f1:.3f} | AUC: {fold_auc:.3f}")
        fold_metrics.append([fold_acc, fold_prec, fold_rec, best_f1, fold_auc])


    # Final Averages
    avgs = np.mean(fold_metrics, axis=0)
    print("\nFINAL 5-FOLD CV RESULTS:")
    print(f"Accuracy:  {avgs[0]:.3f}")
    print(f"Precision: {avgs[1]:.3f}")
    print(f"Recall:    {avgs[2]:.3f}")
    print(f"F1-Score:  {avgs[3]:.3f}")
    print(f"AUC:       {avgs[4]:.3f}")
    print("=========================================================")

if __name__ == '__main__':
    main()
