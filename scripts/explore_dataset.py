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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 15,
        "figure.titlesize": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
    }
)

ROOT        = Path(__file__).resolve().parent.parent
SPECTRAL_DS = ROOT / 'datasets' / 'spectral_dataset.parquet'
LABELS_CSV  = ROOT / 'dataset_statistics_summary.csv'

df = pd.read_parquet(SPECTRAL_DS)
band_columns = [col for col in df.columns if col.startswith('band_')]

# Map each Class_ID to its human-readable Class_Name.
labels = pd.read_csv(LABELS_CSV)
label_names = dict(zip(labels['Class_ID'], labels['Class_Name']))

# %% --- single spectrum ---

row = df.iloc[152990]
spectral_values = row[band_columns]

plt.figure(figsize=(10, 5))
plt.plot(spectral_values.values)
plt.title(f"Spectrum – Sample {row['Sample_ID']} at ({row['x']}, {row['y']}), label={row['Label']}")
plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True)
plt.show()

# %% --- per-label average spectra ---

unique_labels = df['Label'].unique()
avg_spectra   = {}

for label in unique_labels:
    temp_avg = df.loc[df['Label'] == label, band_columns].mean()
    temp_std = df.loc[df['Label'] == label, band_columns].std()
    avg_spectra[label] = (temp_avg, temp_std)

    name = label_names.get(label, 'Unknown')
    plt.figure(figsize=(10, 5))
    x = range(len(temp_avg))
    plt.plot(x, temp_avg.values)
    plt.fill_between(x,
                 (temp_avg - temp_std).values,
                 (temp_avg + temp_std).values,
                 alpha=0.3)
    plt.title(f"Spectrum – label {label} ({name})")
    plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True)
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
for label in [2, 0, 15, -1]: 
    name = label_names.get(label, 'Unknown')
    temp_mean = avg_spectra[label][0].values.mean()
    temp_std = avg_spectra[label][0].values.std()
    plt.plot((avg_spectra[label][0].values - temp_mean)/temp_std , label=f'Label {label} ({name})')
plt.xlabel("Band Index"); plt.ylabel("Intensity"); plt.grid(True); plt.legend()
plt.show()

# %% --- label distribution ---

label_counts = df['Label'].value_counts().sort_values()
# label_counts = label_counts.drop([-1, 15])

# Label each bar with "<id> (<name>)" instead of just the numeric id.
tick_labels = [f"{label_names.get(label, 'Unknown')} - {label} "
               for label in label_counts.index]

plt.figure(figsize=(10, 5))
label_counts.plot(kind='barh')
plt.yticks(range(len(label_counts)), tick_labels, fontsize=18)
plt.xlabel('Number of pixels'); plt.ylabel('')
plt.tight_layout()
plt.show()

# %% --- label 2 vs rest ---

label_counts = label_counts.drop([-1, 15])
count_2   = label_counts.loc[2]
count_rest = label_counts.drop(2).sum()
pd.Series({'label 2': count_2, 'other valid labels': count_rest}).plot(kind='barh',
                                                                        figsize=(10, 5))
plt.xlabel('Number of pixels')
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# %% --- tumoral / healthy / discarded spatial maps ---
TUMOR_LABELS    = [2, 20]     # pixels treated as tumoral
DISCARD_LABELS  = [-1, 15]       # pixels discarded; everything else is healthy

# 0 = discarded, 1 = healthy, 2 = tumoral
cmap = ListedColormap(['white', 'forestgreen', 'darkred'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)


def categorize(label):
    if label in DISCARD_LABELS:
        return 0
    if label in TUMOR_LABELS:
        return 2
    return 1


def plot_category_map(sample_id, map_id, sub):
    sub = sub.copy()
    sub['cat'] = sub['Label'].apply(categorize)
    grid = sub.pivot_table(index='y', columns='x', values='cat', aggfunc='first')

    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Discarded', 'Healthy', 'Tumoral'], fontsize=18)
    plt.tight_layout(); plt.show()


for (sample_id, map_id), sub in df.groupby(['Sample_ID', 'Map_ID']):
    plot_category_map(sample_id, map_id, sub)

# %% --- tumoral macroclass vs single other class spatial maps ---

# Pick the tumoral macroclass and a single other class to contrast against it.
TUMOR_MACRO     = [2,20]     # pixels treated as tumoral
OTHER_CLASS     = 6              # the single class to distinguish from tumoral
INVALID_LABELS  = [-1, 15]       # pixels treated as invalid (shown white)

# 0 = invalid (white), 1 = other valid (gray), 2 = chosen (yellow), 3 = tumoral
cmap_bin = ListedColormap(['white', 'lightgray', 'yellow', 'darkred'])
cmap_bin.set_bad('white')   # pixel assenti dal grid -> bianco
norm_bin = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_bin.N)


def categorize_binary(label):
    if label in INVALID_LABELS:
        return 0    # no tissue -> white
    if label in TUMOR_MACRO:
        return 3
    if label == OTHER_CLASS:
        return 2
    return 1        # other valid tissue -> gray


def plot_binary_map(sample_id, map_id, sub):
    sub = sub.copy()
    sub['cat'] = sub['Label'].apply(categorize_binary)
    grid = sub.pivot_table(index='y', columns='x', values='cat', aggfunc='first')

    other_name = label_names.get(OTHER_CLASS, 'Unknown')
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap=cmap_bin, norm=norm_bin, origin='lower')
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Discarded', 'Other', other_name , 'Tumoral'], fontsize=18)
    plt.tight_layout(); plt.show()


for (sample_id, map_id), sub in df.groupby(['Sample_ID', 'Map_ID']):
    plot_binary_map(sample_id, map_id, sub)

# %% --- samples containing a chosen class ---

CHOSEN_CLASS = 5    # the class to look for across samples

samples_with_class = df.loc[df['Label'] == CHOSEN_CLASS, 'Sample_ID'].unique()

name = label_names.get(CHOSEN_CLASS, 'Unknown')
print(f"Class {CHOSEN_CLASS} ({name}) appears in {len(samples_with_class)} samples:")
for sid in samples_with_class:
    print(f"  {sid}")

