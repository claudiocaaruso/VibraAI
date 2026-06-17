"""
Exploratory data analysis on the raw Raman spectral dataset.

Visualises individual spectra, spatial heatmaps, per-label average spectra,
and label distributions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT        = Path(__file__).resolve().parent.parent
SPECTRAL_DS = ROOT / 'datasets' / 'spectral_dataset.parquet'
LABELS_CSV  = ROOT / 'dataset_statistics_summary.csv'

df = pd.read_parquet(SPECTRAL_DS)
band_columns = [col for col in df.columns if col.startswith('band_')]

# Map each Class_ID to its human-readable Class_Name.
labels = pd.read_csv(LABELS_CSV)
label_names = dict(zip(labels['Class_ID'], labels['Class_Name']))

# %% --- single spectrum ---

row = df.iloc[6000]
spectral_values = row[band_columns]

plt.figure(figsize=(10, 5))
plt.plot(spectral_values.values)
plt.title(f"Spectrum – Sample {row['Sample_ID']} at ({row['x']}, {row['y']}), label={row['Label']}")
plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True)
plt.show()

# %% --- spatial heatmap for one band ---

target_band = 'band_341'
sample_id   = df['Sample_ID'].unique()[2]
sample_data = df[df['Sample_ID'] == sample_id]
heatmap_data = sample_data.pivot_table(index='y', columns='x', values=target_band)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title(f"Map of {target_band} – Sample {sample_id}")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()

# %% --- per-label average spectra ---

unique_labels = df['Label'].unique()
avg_spectra   = {}

for label in unique_labels:
    temp_avg = df.loc[df['Label'] == label, band_columns].mean()
    temp_std = df.loc[df['Label'] == label, band_columns].std()
    avg_spectra[label] = (temp_avg, temp_std)

    name = label_names.get(label, 'Unknown')
    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg)
    plt.fill_between(range(len(band_columns)),
                     temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"Spectrum – label {label} ({name})")
    plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(False)
    plt.show()

# %% --- normalised spectra comparison ---

plt.figure(figsize=(10, 5))
for label in unique_labels[~np.isin(unique_labels, [-1])]:  #removing unlabeled spectra (too different)
    peak = max(avg_spectra[label][0])
    name = label_names.get(label, 'Unknown')
    plt.plot(avg_spectra[label][0].values / peak, label=f'Label {label} ({name})')
plt.title("Normalised spectra comparison")
plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True); plt.legend()
plt.show()

# %% --- normalised spectra comparison with manual spectra selection ---

plt.figure(figsize=(10, 5))
for label in [2, 6, 4, 5, 20, 0, 15]: 
    name = label_names.get(label, 'Unknown')
    temp_mean = avg_spectra[label][0].values.mean()
    temp_std = avg_spectra[label][0].values.std()
    plt.plot((avg_spectra[label][0].values - temp_mean)/temp_std , label=f'Label {label} ({name})')
plt.title("Normalised spectra comparison")
plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True); plt.legend()
plt.show()

# %% --- label distribution ---

label_counts = df['Label'].value_counts().sort_values()
label_counts = label_counts.drop([-1, 15])

plt.figure(figsize=(10, 5))
label_counts.plot(kind='barh')
plt.ylabel('Label'); plt.xlabel('Number of pixels')
plt.show()

# %% --- label 2 vs rest ---

count_2   = label_counts.loc[2]
count_rest = label_counts.drop(2).sum()
pd.Series({'label 2': count_2, 'other valid labels': count_rest}).plot(kind='barh',
                                                                        figsize=(10, 5))
plt.ylabel('Label'); plt.xlabel('Number of pixels')
plt.show()

# %% --- tumoral / healthy / discarded spatial maps ---
sample_IDs = np.append(df['Sample_ID'].unique(), None)
TUMOR_LABELS    = [2, 20, 0]     # pixels treated as tumoral
DISCARD_LABELS  = [-1, 15]       # pixels discarded; everything else is healthy
SELECTED_SAMPLE = sample_IDs[-1]           # a Sample_ID to plot just one, or [-1] to loop over all

# 0 = discarded, 1 = healthy, 2 = tumoral
cmap = ListedColormap(['lightgray', 'forestgreen', 'red'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)


def categorize(label):
    if label in DISCARD_LABELS:
        return 0
    if label in TUMOR_LABELS:
        return 2
    return 1


def plot_category_map(sample_id):
    sub = df[df['Sample_ID'] == sample_id].copy()
    sub['cat'] = sub['Label'].apply(categorize)
    grid = sub.pivot_table(index='y', columns='x', values='cat', aggfunc='first')

    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Discarded', 'Healthy', 'Tumoral'])
    plt.title(f"Sample {sample_id} – tumoral / healthy / discarded")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.show()


if SELECTED_SAMPLE is not None:
    plot_category_map(SELECTED_SAMPLE)
else:
    for sid in df['Sample_ID'].unique():
        plot_category_map(sid)

