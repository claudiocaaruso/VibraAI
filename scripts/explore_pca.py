"""
PCA on the raw Raman spectral dataset.

Fits a full PCA, plots cumulative explained variance, builds the
X_pca_dataset.parquet used for training, and visualises per-label
projections and reconstructed spectra.
The parquet save commands at the bottom are commented out by default —
uncomment and run once to regenerate the PCA dataset.
"""
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT        = Path(__file__).resolve().parent.parent
SPECTRAL_DS = ROOT / 'datasets' / 'spectral_dataset.parquet'

df = pd.read_parquet(SPECTRAL_DS)
band_columns = [col for col in df.columns if col.startswith('band_')]

# %% --- create a dataset without useless labels (-1, 15) ---

df = df[~df['Label'].isin([-1, 15])]
gc.collect()

X = df[band_columns].to_numpy()
print(f"X shape: {X.shape}")

# %% --- SNV normalisation ---

X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# %% --- per-label average spectra on SNV data (needed for reconstruction comparison) ---
explained_var = 0.90

y           = df['Label'].values
labels      = np.unique(y)
avg_spectra = {}

for label in labels:
    mask = y == label
    avg_spectra[label] = (X[mask].mean(axis=0), X[mask].std(axis=0))

pca     = PCA(n_components=None)
pca.fit(X)
X_pca   = pca.transform(X)
Vt      = pca.components_
cumvar  = np.cumsum(pca.explained_variance_ratio_)

req_dims = np.argmax(cumvar >= explained_var) + 1
print(f"PCs needed for {explained_var*100}% explained variance: {req_dims}")

plt.figure(figsize=(10, 5))
plt.plot(cumvar, label='Cumulative variance')
plt.axvline(x=req_dims, color='r', linestyle='--', label=f'{req_dims} PCs')
plt.axhline(y=explained_var,     color='r', linestyle='--', label=f'{explained_var*100}% threshold')
plt.title("Cumulative explained variance")
plt.xlabel("Dimensions"); plt.ylabel("Explained variance")
plt.legend(); plt.grid(True); plt.show()

# %% --- build X_pca dataframe ---

X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
X_pca_df['Label'] = df['Label'].values
X_pca_df = pd.concat([
    df[['Sample_ID', 'Map_ID', 'x', 'y']].reset_index(drop=True),
    X_pca_df,
], axis=1)
print(X_pca_df.head())

# Uncomment to regenerate the PCA dataset:
# X_pca_df.to_parquet(ROOT / 'datasets' / 'X_pca_dataset.parquet')
# pd.DataFrame(Vt).to_parquet(ROOT / 'datasets' / 'Vt_dataset.parquet')

# %% --- per-label average in PC space ---

n_PCs      = 40
PC_columns = [f'PC{i+1}' for i in range(n_PCs)]

for label in labels:
    temp_avg = X_pca_df.loc[X_pca_df['Label'] == label, PC_columns].mean()
    temp_std = X_pca_df.loc[X_pca_df['Label'] == label, PC_columns].std()
    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg)
    plt.fill_between(range(1, n_PCs + 1), temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"PC-space spectrum – label {label} ({n_PCs} PCs)")
    plt.xlabel("PC Index"); plt.ylabel("Projection"); plt.grid(False)
    plt.show()

# %% --- reconstructed vs original spectra ---

n_PCs      = 2
Vt_reduced = Vt[:n_PCs, :]

for label in labels:
    mask     = y == label
    X_rec    = X_pca[mask, :n_PCs] @ Vt_reduced + pca.mean_
    temp_avg = X_rec.mean(axis=0)
    temp_std = X_rec.std(axis=0)
    orig_avg = avg_spectra[label][0]

    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg,  label='Reconstructed')
    plt.plot(orig_avg,  label='Original', linestyle='--')
    plt.fill_between(range(len(band_columns)), temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"Reconstructed vs original – label {label} (k={n_PCs})")
    plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.legend(); plt.grid(False)
    plt.show()

    del X_rec; gc.collect()
