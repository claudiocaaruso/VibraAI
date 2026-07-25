import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
    }
)

FIGSIZE = (9, 5)
SCATTER_FIGSIZE = (8, 6)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PCA_DIR = os.path.join(ROOT_DIR, "PCA")
PCA_IMAGES_DIR = os.path.join(ROOT_DIR, "pca_images")
os.makedirs(PCA_IMAGES_DIR, exist_ok=True)

DATASET_PATH = os.path.join(PCA_DIR, "spectral_dataset_clean.parquet")
PCA_DATASET_PATH = os.path.join(PCA_DIR, "spectral_dataset_pca_483.parquet")
PCA_COMPONENTS_NPY_PATH = os.path.join(PCA_DIR, "pca_components_483.npy")
PCA_COMPONENTS_PARQUET_PATH = os.path.join(PCA_DIR, "pca_components_483.parquet")

SELECTED_ROW_INDEX = 205430
SELECTED_BAND = "band_341"


def save_current_figure(filename):
    output_path = os.path.join(PCA_IMAGES_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


# %%
# Load the cleaned dataset.
df = pd.read_parquet(DATASET_PATH)

band_columns = [col for col in df.columns if col.startswith("band_")]
metadata_columns = ["Sample_ID", "Map_ID", "x", "y"]
label_column = "Label"

X = df[band_columns]
selected_row = df.iloc[SELECTED_ROW_INDEX]
selected_sample_id = selected_row["Sample_ID"]

print("Dataset preview:")
print(df.head(20))
print("Spectral matrix X shape:")
print(X.shape)
print("Number of Raman bands:")
print(len(band_columns))


# %%
# Plot one original Raman spectrum.
row_index = SELECTED_ROW_INDEX
row = selected_row
spectral_values = row[band_columns]

plt.figure(figsize=FIGSIZE)
plt.plot(spectral_values.values)
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.grid(True)
plt.tight_layout()
save_current_figure("01_original_raman_spectrum.png")
plt.show()

# %%
# Plot the spatial map of one selected band for one sample.
target_band = SELECTED_BAND
sample_id = selected_sample_id
sample_data = df[df["Sample_ID"] == sample_id]
heatmap_data = sample_data.pivot_table(index="y", columns="x", values=target_band)

plt.figure(figsize=SCATTER_FIGSIZE)
plt.imshow(heatmap_data, cmap="viridis", origin="lower")
plt.colorbar(label="Intensity")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.tight_layout()
save_current_figure("02_spatial_map_selected_band.png")
plt.show()


# %%
# Mean-centering by hand.
mu = X.mean(axis=0)
X_centered_manual = X - mu

print("Mean spectrum shape:")
print(mu.shape)
print("Largest residual column mean after manual centering:")
print(X_centered_manual.mean(axis=0).abs().max())

plt.figure(figsize=FIGSIZE)
plt.plot(mu.values)
plt.xlabel("Band Index")
plt.ylabel("Mean intensity")
plt.grid(True)
plt.tight_layout()
save_current_figure("03_mean_raman_spectrum.png")
plt.show()


# %%
# Mean-centering with Scikit-Learn.
# with_std=False subtracts the mean without dividing by the standard deviation.
scaler = StandardScaler(with_mean=True, with_std=False)
X_centered_sklearn = scaler.fit_transform(X)

print("Mean spectrum from StandardScaler shape:")
print(scaler.mean_.shape)
print("Largest residual column mean after StandardScaler centering:")
print(np.abs(X_centered_sklearn.mean(axis=0)).max())


# %%
# Check that the two centering methods give the same result.
max_difference = np.abs(X_centered_manual.to_numpy() - X_centered_sklearn).max()

print("Max difference between manual and StandardScaler centering:")
print(max_difference)


# %%
# Compare the first centered spectrum from the two methods.
row_index = SELECTED_ROW_INDEX
first_spectrum_manual = X_centered_manual.iloc[row_index]
first_spectrum_sklearn = X_centered_sklearn[row_index]

plt.figure(figsize=FIGSIZE)
plt.plot(first_spectrum_manual.values, label="Manual centering")
plt.plot(first_spectrum_sklearn, linestyle="--", label="StandardScaler centering")
plt.xlabel("Band Index")
plt.ylabel("Centered intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_figure("04_centering_manual_vs_sklearn.png")
plt.show()

# %%
# Compare the original spectrum with the centered spectrum.
original_spectrum = X.iloc[row_index]
centered_spectrum = X_centered_manual.iloc[row_index]

plt.figure(figsize=FIGSIZE)
plt.plot(original_spectrum.values, label="Original spectrum")
plt.plot(centered_spectrum.values, label="Centered spectrum")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_figure("05_original_vs_centered_spectrum.png")
plt.show()


# %%
# Fit PCA without reducing the number of components.
# Scikit-Learn centers X internally before computing PCA.
pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1

print(f"Number of components needed to preserve 95% of the variance: {n_components_95}")
print(f"Explained variance with {n_components_95} components: {cumsum[n_components_95 - 1]:.4f}")

plt.figure(figsize=FIGSIZE)
plt.plot(cumsum, linewidth=2)
plt.axhline(y=0.95, color="red", linestyle="--", label="95% variance")
plt.axvline(
    x=n_components_95 - 1,
    color="green",
    linestyle="--",
    label=f"{n_components_95} components",
)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_figure("06_cumulative_explained_variance.png")
plt.show()


# %%
# Scree plot: percentage of variance explained by each principal component.
n_components_scree_plot = 50
component_numbers = np.arange(1, n_components_scree_plot + 1)
explained_variance_percent = (
    pca.explained_variance_ratio_[:n_components_scree_plot] * 100
)

plt.figure(figsize=FIGSIZE)
plt.bar(component_numbers, explained_variance_percent, color="steelblue", alpha=0.9)
plt.plot(
    component_numbers,
    explained_variance_percent,
    color="black",
    marker="o",
    linewidth=2,
)
plt.xlabel("Principal Components")
plt.ylabel("Percentage of explained variance")
plt.xticks([1] + list(range(5, n_components_scree_plot + 1, 5)))
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
save_current_figure("07_scree_plot_explained_variance.png")
plt.show()


# %%
# Transform every spectrum into PCA coordinates.
# X_centered = X_pca @ pca.components_
X_pca = pca.transform(X)
pc_columns = [f"pc_{i + 1}" for i in range(X_pca.shape[1])]

X_pca_df = pd.DataFrame(
    X_pca.astype(np.float32, copy=False),
    columns=pc_columns,
    index=df.index,
)

df_pca = pd.concat([df[metadata_columns], X_pca_df, df[[label_column]]], axis=1)

print("Original matrix X shape:")
print(X.shape)
print("Mean spectrum pca.mean_ shape:")
print(pca.mean_.shape)
print("Principal directions pca.components_ shape:")
print(pca.components_.shape)
print("PCA coordinates X_pca shape:")
print(X_pca.shape)
print("PCA dataframe with metadata and labels shape:")
print(df_pca.shape)
print("First rows of df_pca:")
print(df_pca.head())
print("Label counts are preserved:")
print(
    df[label_column].value_counts().sort_index().equals(
        df_pca[label_column].value_counts().sort_index()
    )
)


# %%
# Inspect the PCA decomposition.
print("First 10 values of the mean spectrum:")
print(pca.mean_[:10])
print("First 10 weights of PC1:")
print(pca.components_[0, :10])
print("First 10 weights of PC2:")
print(pca.components_[1, :10])
print("Coordinates of the first spectrum on the first 5 PCs:")
print(X_pca[0, :5])
print("First 10 explained variance ratios:")
print(pca.explained_variance_ratio_[:10])
print("First 10 singular values:")
print(pca.singular_values_[:10])


# %%
# Save the full PCA representation with 483 components.
# The pc_* columns are PCA coordinates, not original Raman bands.
pca_full_dataset = df_pca
pca_full_dataset.to_parquet(PCA_DATASET_PATH, index=False)

np.save(PCA_COMPONENTS_NPY_PATH, pca.components_)

pca_components_df = pd.DataFrame(
    pca.components_.astype(np.float32, copy=False),
    columns=band_columns,
    index=pc_columns,
)
pca_components_df.index.name = "component"
pca_components_df.to_parquet(PCA_COMPONENTS_PARQUET_PATH)

print(f"Saved PCA 483 dataset to: {PCA_DATASET_PATH}")
print(f"Saved PCA components to: {PCA_COMPONENTS_NPY_PATH}")
print(f"Saved PCA components parquet to: {PCA_COMPONENTS_PARQUET_PATH}")
print("PCA 483 dataset shape:")
print(pca_full_dataset.shape)
print("PCA components shape:")
print(pca.components_.shape)
print("Label counts are preserved in the saved dataset:")
print(
    df[label_column].value_counts().sort_index().equals(
        pca_full_dataset[label_column].value_counts().sort_index()
    )
)
print("Variance preserved with 483 components:")
print(pca.explained_variance_ratio_.sum())


# %%
# Read the saved files and print a preview.
saved_pca_dataset = pd.read_parquet(PCA_DATASET_PATH)
saved_pca_components = pd.read_parquet(PCA_COMPONENTS_PARQUET_PATH)

print("Saved PCA dataset preview:")
print(saved_pca_dataset.head())
print("Saved PCA dataset shape:")
print(saved_pca_dataset.shape)
print("Saved PCA dataset columns:")
print(saved_pca_dataset.columns.tolist())

print("Saved PCA components preview:")
print(saved_pca_components.head())
print("Saved PCA components shape:")
print(saved_pca_components.shape)
print("Saved PCA components columns:")
print(saved_pca_components.columns.tolist())


# %%
# Scatter plot of the spectra projected onto the first two principal components.
# A random sample is used to keep the plot readable.
n_points_pca_scatter = 10000

if len(saved_pca_dataset) > n_points_pca_scatter:
    pca_scatter_df = saved_pca_dataset.sample(n=n_points_pca_scatter, random_state=42)
else:
    pca_scatter_df = saved_pca_dataset

plt.figure(figsize=SCATTER_FIGSIZE)

labels = sorted(pca_scatter_df[label_column].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

for label, color in zip(labels, colors):
    label_data = pca_scatter_df[pca_scatter_df[label_column] == label]
    plt.scatter(
        label_data["pc_1"],
        label_data["pc_2"],
        color=color,
        s=8,
        alpha=0.6,
        label=f"Label {label}",
    )

plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.4)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(
    title="Label",
    markerscale=2,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_current_figure("08_pca_score_plot_pc1_pc2.png")
plt.show()


# %%
# Scatter plot between two selected principal components.
# Change these values to choose which components to compare.
pc_x_index = 1
pc_y_index = 483
pc_x_column = f"pc_{pc_x_index}"
pc_y_column = f"pc_{pc_y_index}"

plt.figure(figsize=SCATTER_FIGSIZE)

for label, color in zip(labels, colors):
    label_data = pca_scatter_df[pca_scatter_df[label_column] == label]
    plt.scatter(
        label_data[pc_x_column],
        label_data[pc_y_column],
        color=color,
        s=8,
        alpha=0.6,
        label=f"Label {label}",
    )

plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.4)
plt.xlabel(f"PC{pc_x_index}")
plt.ylabel(f"PC{pc_y_index}")
plt.legend(
    title="Label",
    markerscale=2,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_current_figure(f"09_pca_score_plot_pc{pc_x_index}_pc{pc_y_index}.png")
plt.show()


# %%
# Plot one spectrum represented in the PCA space with all 483 components.
row_index = SELECTED_ROW_INDEX
pca_spectrum_483 = saved_pca_dataset.loc[row_index, pc_columns].to_numpy()
row_metadata = saved_pca_dataset.loc[row_index, metadata_columns + [label_column]]

plt.figure(figsize=FIGSIZE)
plt.plot(range(1, len(pca_spectrum_483) + 1), pca_spectrum_483)
plt.xlabel("Principal component index")
plt.ylabel("PCA coordinate")
plt.grid(True)
plt.tight_layout()
save_current_figure("10_pca_space_spectrum_483_components.png")
plt.show()

# %%
# Plot the same PCA-space spectrum after keeping only the first components.
n_components_to_plot = 300
reduced_pc_columns = [f"pc_{i + 1}" for i in range(n_components_to_plot)]
pca_spectrum_reduced = saved_pca_dataset.loc[row_index, reduced_pc_columns].to_numpy()

plt.figure(figsize=FIGSIZE)
plt.plot(range(1, n_components_to_plot + 1), pca_spectrum_reduced)
plt.xlabel("Principal component index")
plt.ylabel("PCA coordinate")
plt.grid(True)
plt.tight_layout()
save_current_figure(f"11_pca_space_spectrum_{n_components_to_plot}_components.png")
plt.show()


# %%
# Reconstruct the original Raman spectrum using selected numbers of PCA components.
original_spectrum = X.iloc[row_index].values

for n_components_reconstruction in [1, 89]:
    X_reconstructed = (
        X_pca[:, :n_components_reconstruction]
        @ pca.components_[:n_components_reconstruction, :]
        + pca.mean_
    )
    reconstructed_spectrum = X_reconstructed[row_index]

    plt.figure(figsize=FIGSIZE)
    plt.plot(original_spectrum, label="Original spectrum")
    plt.plot(
        reconstructed_spectrum,
        linestyle="--",
        label=f"Reconstructed with {n_components_reconstruction} PCs",
    )
    plt.xlabel("Band Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_current_figure(
        f"12_original_vs_reconstructed_{n_components_reconstruction}_components.png"
    )
    plt.show()
