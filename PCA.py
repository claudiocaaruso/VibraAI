import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "/Users/giuliatarsi/Desktop/VIBRA/spectral_dataset_clean.parquet"
PCA_DATASET_PATH = "/Users/giuliatarsi/Desktop/VIBRA/spectral_dataset_pca_483.parquet"
PCA_COMPONENTS_NPY_PATH = "/Users/giuliatarsi/Desktop/VIBRA/pca_components_483.npy"
PCA_COMPONENTS_PARQUET_PATH = "/Users/giuliatarsi/Desktop/VIBRA/pca_components_483.parquet"


# %%
# Load the cleaned dataset.
df = pd.read_parquet(DATASET_PATH)

band_columns = [col for col in df.columns if col.startswith("band_")]
metadata_columns = ["Sample_ID", "Map_ID", "x", "y"]
label_column = "Label"

X = df[band_columns]

print("Dataset preview:")
print(df.head(20))
print("Spectral matrix X shape:")
print(X.shape)
print("Number of Raman bands:")
print(len(band_columns))


# %%
# Plot one original Raman spectrum.
row_index = 205430
row = df.iloc[row_index]
spectral_values = row[band_columns]

plt.figure(figsize=(10, 5))
plt.plot(spectral_values.values)
plt.title(f"Spectrum for Sample {row['Sample_ID']} at ({row['x']}, {row['y']})")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot the spatial map of one selected band for one sample.
target_band = "band_341"
sample_id = df["Sample_ID"].unique()[2]
sample_data = df[df["Sample_ID"] == sample_id]
heatmap_data = sample_data.pivot_table(index="y", columns="x", values=target_band)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap="viridis", origin="lower")
plt.colorbar(label="Intensity")
plt.title(f"Map of {target_band} for Sample {sample_id}")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.tight_layout()
plt.show()


# %%
# Mean-centering by hand.
mu = X.mean(axis=0)
X_centered_manual = X - mu

print("Mean spectrum shape:")
print(mu.shape)
print("Largest residual column mean after manual centering:")
print(X_centered_manual.mean(axis=0).abs().max())

plt.figure(figsize=(10, 5))
plt.plot(mu.values)
plt.title("Mean Raman spectrum")
plt.xlabel("Band Index")
plt.ylabel("Mean intensity")
plt.grid(True)
plt.tight_layout()
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
row_index = 0
first_spectrum_manual = X_centered_manual.iloc[row_index]
first_spectrum_sklearn = X_centered_sklearn[row_index]

plt.figure(figsize=(10, 5))
plt.plot(first_spectrum_manual.values, label="Manual centering")
plt.plot(first_spectrum_sklearn, linestyle="--", label="StandardScaler centering")
plt.title(f"Centered Raman spectrum for row {row_index}")
plt.xlabel("Band Index")
plt.ylabel("Centered intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Compare the original spectrum with the centered spectrum.
original_spectrum = X.iloc[row_index]
centered_spectrum = X_centered_manual.iloc[row_index]

plt.figure(figsize=(10, 5))
plt.plot(original_spectrum.values, label="Original spectrum")
plt.plot(centered_spectrum.values, label="Centered spectrum")
plt.title(f"Original vs centered Raman spectrum for row {row_index}")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
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

plt.figure(figsize=(10, 5))
plt.plot(cumsum, linewidth=2)
plt.axhline(y=0.95, color="red", linestyle="--", label="95% variance")
plt.axvline(
    x=n_components_95 - 1,
    color="green",
    linestyle="--",
    label=f"{n_components_95} components",
)
plt.title("Cumulative explained variance")
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Scree plot: percentage of variance explained by each principal component.
n_components_scree_plot = 50
component_numbers = np.arange(1, n_components_scree_plot + 1)
explained_variance_percent = (
    pca.explained_variance_ratio_[:n_components_scree_plot] * 100
)

plt.figure(figsize=(10, 5))
plt.bar(component_numbers, explained_variance_percent, color="steelblue", alpha=0.9)
plt.plot(
    component_numbers,
    explained_variance_percent,
    color="black",
    marker="o",
    linewidth=2,
)
plt.title("Explained variance by principal component")
plt.xlabel("Principal Components")
plt.ylabel("Percentage of explained variance")
plt.xticks(component_numbers)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
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

plt.figure(figsize=(9, 7))

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
plt.title("Spectra projected onto the first two principal components")
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
plt.show()


# %%
# Scatter plot between two selected principal components.
# Change these values to choose which components to compare.
pc_x_index = 1
pc_y_index = 483
pc_x_column = f"pc_{pc_x_index}"
pc_y_column = f"pc_{pc_y_index}"

plt.figure(figsize=(9, 7))

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
plt.title(f"Spectra projected onto PC{pc_x_index} and PC{pc_y_index}")
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
plt.show()


# %%
# Plot one spectrum represented in the PCA space with all 483 components.
row_index = 30
pca_spectrum_483 = saved_pca_dataset.loc[row_index, pc_columns].to_numpy()
row_metadata = saved_pca_dataset.loc[row_index, metadata_columns + [label_column]]

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca_spectrum_483) + 1), pca_spectrum_483)
plt.title(
    f"PCA-space spectrum for Sample {row_metadata['Sample_ID']} "
    f"at ({row_metadata['x']}, {row_metadata['y']}), Label {row_metadata['Label']}"
)
plt.xlabel("Principal component index")
plt.ylabel("PCA coordinate")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot the same PCA-space spectrum after keeping only the first components.
n_components_to_plot = 300
reduced_pc_columns = [f"pc_{i + 1}" for i in range(n_components_to_plot)]
pca_spectrum_reduced = saved_pca_dataset.loc[row_index, reduced_pc_columns].to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_components_to_plot + 1), pca_spectrum_reduced)
plt.title(
    f"PCA-space spectrum with {n_components_to_plot} components "
    f"for Sample {row_metadata['Sample_ID']} "
    f"at ({row_metadata['x']}, {row_metadata['y']}), Label {row_metadata['Label']}"
)
plt.xlabel("Principal component index")
plt.ylabel("PCA coordinate")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Reconstruct the original Raman spectrum using only the first PCA components.
n_components_reconstruction = 1
X_reconstructed = (
    X_pca[:, :n_components_reconstruction]
    @ pca.components_[:n_components_reconstruction, :]
    + pca.mean_
)

original_spectrum = X.iloc[row_index].values
reconstructed_spectrum = X_reconstructed[row_index]
row_label = df[label_column].iloc[row_index]

plt.figure(figsize=(10, 5))
plt.plot(original_spectrum, label="Original spectrum")
plt.plot(
    reconstructed_spectrum,
    linestyle="--",
    label=f"Reconstructed with {n_components_reconstruction} PCs",
)
plt.title(f"Original vs reconstructed spectrum, row {row_index}, Label {row_label}")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
