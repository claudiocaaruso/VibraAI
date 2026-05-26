import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/claudiocaruso/Desktop/ingegneria Fisica/Tesi Vibra AI/spectral_dataset.parquet"

df = pd.read_parquet(path)


# %%


row = df.iloc[60]

band_columns = [col for col in df.columns if col.startswith('band_')]
spectral_values = row[band_columns]

plt.figure(figsize=(10, 5))
plt.plot(spectral_values.values)
plt.title(f"Spectrum for Sample {row['Sample_ID']} at ({row['x']}, {row['y']}), label = {row['Label']}")
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

labels = df['Label'].unique()           # serie contenente tutte le labels di df
avg_spectra = {}                        # dict per associare le labels a serie di spettri medi e std

for label in labels:                    # riempio il dict avg_spectra
    temp_avg = df.loc[df['Label'] == label , band_columns].mean()   
    temp_std = df.loc[df['Label'] == label , band_columns].std()
    avg_spectra[label] = (temp_avg, temp_std)

    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg)
    plt.fill_between(range(len(band_columns)), temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"Spectrum for label {label}")
    plt.xlabel("Band Index")
    plt.ylabel("Intensity")
    plt.grid(False)
    plt.show()
    
#%%

plt.figure(figsize=(10, 5))

for label in labels:
    peak = max(avg_spectra[label][0])       # valuto il picco degli spettri
    plt.plot((avg_spectra[label][0].values)/peak, label=f'Label {label}')
    
plt.title("normalized spectra comparison")
plt.xlabel("Band Index")
plt.ylabel("Intensity")
plt.grid(True)
plt.legend()
plt.show()


#%%

label_counts = df['Label'].value_counts().sort_values()
label_counts = label_counts.drop([-1,15])

plt.figure(figsize=(10, 5))
label_counts.plot(kind='barh')
plt.ylabel('labels')
plt.xlabel('number of pixels')
plt.show()

#%%

count_2 = label_counts.loc[2]
count_rest = label_counts.drop(2).sum()
comparison = pd.Series({'label 2':count_2, 'other valid labels':count_rest})

plt.figure(figsize=(10, 5))
comparison.plot(kind='barh')
plt.ylabel('labels')
plt.xlabel('number of pixels')
plt.show()

#%%


    
