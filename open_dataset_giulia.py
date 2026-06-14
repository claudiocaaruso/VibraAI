import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/giuliatarsi/Desktop/VIBRA/PCA/spectral_dataset.parquet"

df = pd.read_parquet(path)

print(df.head(20))

# %%

df.iloc[1]

row = df.iloc[205430]

band_columns = [col for col in df.columns if col.startswith('band_')]
spectral_values = row[band_columns]

plt.figure(figsize=(10, 5))
plt.plot(spectral_values.values)
plt.title(f"Spectrum for Sample {row['Sample_ID']} at ({row['x']}, {row['y']})")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()

# %%

target_band = 'band_341'

sample_id = df['Sample_ID'].unique()[2]

sample_data = df[df['Sample_ID'] == sample_id]

heatmap_data = sample_data.pivot_table(index='y', columns='x', values=target_band)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title(f"Map of {target_band} for Sample {sample_id}")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.tight_layout()
plt.show()

# %%
band_columns = [col for col in df.columns if col.startswith('band_')]

avg = {}

for label in df['Label'].unique():
    avg[label] = df[df['Label'] == label][band_columns].mean()

plt.figure(figsize=(10,5))

for label in avg:
    plt.plot(avg[label].values, label=label)

plt.legend()
plt.grid(True)
plt.xlabel("Band Index")
plt.ylabel("Average Intensity")
plt.title("Average Raman spectra by label")

plt.show()

# %%
band_columns = [col for col in df.columns if col.startswith('band_')]

labels_to_plot = [0, 2, 6, 20]

avg = {}

for label in labels_to_plot:
    avg[label] = df[df['Label'] == label][band_columns].mean()

plt.figure(figsize=(10, 5))

for label in labels_to_plot:
    plt.plot(avg[label].values, label=f"Label {label}")

plt.legend()
plt.grid(True)
plt.xlabel("Band Index")
plt.ylabel("Average Intensity")
plt.title("Average Raman spectra for labels 0, 2, 6, 20")

plt.show()
#%%

plt.figure(figsize=(10, 5))
label_counts = df['Label'].value_counts().sort_values()
label_counts.drop([-1,15]).plot(kind='barh')
plt.show()
#%%
plt.figure(figsize=(10, 5))

# count pixels with label 2
count_2 = (df['Label'] == 2).sum()

# count pixels with label different from 2
count_not_2 = (
    (df['Label'] != 2) &
    (df['Label'] != -1) &
    (df['Label'] != 15)
).sum()


counts = {
    'Label 2': count_2,
    'Not label 2': count_not_2
}

plt.barh(counts.keys(), counts.values())

plt.xlabel("Count")
plt.title("Label 2 vs Not Label 2")

plt.show()

# =============================================================================
# #%%
# =============================================================================
band_columns = [col for col in df.columns if col.startswith('band_')]

label = 2

avg = {}

# save all spectra with the chosen label
avg[label] = df[df['Label'] == label][band_columns]

# compute mean spectrum and standard deviation
mean_spectrum = avg[label].mean()
std_spectrum = avg[label].std()

# plot
plt.figure(figsize=(10,5))

# mean spectrum
plt.plot(mean_spectrum.values, label=f'Label {label}')

# standard deviation region
plt.fill_between(
    range(len(mean_spectrum)),
    mean_spectrum.values - std_spectrum.values,
    mean_spectrum.values + std_spectrum.values,
    alpha=0.4,
    label='±1 std'
)

plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.title(f"Average Raman spectrum for label {label}")

plt.legend()
plt.grid(True)

plt.show()



