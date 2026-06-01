import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.decomposition import PCA


path = "/Users/claudiocaruso/Projects/Tesi Vibra AI/spectral_dataset.parquet"

df = pd.read_parquet(path)


# %%


row = df.iloc[6000]

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

#%% Create a df without labels -1 and 15

df_clean = df[~df['Label'].isin([-1, 15])]
del df
gc.collect()


# %% PCA starts from the spectral matrix X: one row per spectrum, one column per band.

X = df_clean[band_columns]

print(f"X shape: {X.shape}")


# %%
# Mean-centering by hand.

mu = X.mean(axis=0)  # mu is the mean spectrum: one mean value for each Raman band
X_centered_manual = X - mu # a ogni riga di X sottraggo lo spettro medio

print("Mean spectrum shape:", mu.shape) #verifico che mu è un vettore lungo 483
print("Largest residual column mean after manual centering:")
print(X_centered_manual.mean(axis=0).abs().max()) #ricontrolla la media delle colonne di X_centered_manual e printa il valore massimo per verificare che sia molto vicino a zero

# Plot the mean spectrum that is subtracted from every spectrum.
plt.figure(figsize=(10, 5))
plt.plot(mu.values)
plt.title("Mean Raman spectrum")
plt.xlabel("Band Index")
plt.ylabel("Mean intensity")
plt.grid(True)
plt.show()


#%% Riduzione in PCs e somma cumulativa delle explained variances

d = 0                                               # dimensioni finali
pca = PCA(n_components = None if d == 0 else d)     # pca è un oggetto vuoto che ridurrà al numero di PC al suo argomento 
pca.fit(X)                                          # pca viene modificata e diventa Vt con altri dati utili annessi
Vt = pca.components_                                # pca.components_ equivale a Vt
exp_variance = pca.explained_variance_ratio_        # array con le exp variances
X_pca = pca.transform(X)                            # X_pca è la nuova matrice (U ∑) del dataset ridotta a d PCs (vettori orizzontali)

cumvar = np.cumsum(exp_variance)

req_variance = 0.95             # variabile richiesta di explained variance

req_dimensions = np.argmax(cumvar >= req_variance) + 1      # numero di PCs richieste

print(f'num of PCs for {req_variance} explained variance is {req_dimensions}')

plt.figure(figsize=(10, 5))
plt.plot(cumvar, label='cumulative variance')
plt.axvline(x=req_dimensions, color='r', linestyle='--', label=f'{req_dimensions} PCs') # vertical line
plt.axhline(y=req_variance, color='r', linestyle='--', label=f'{req_variance*100}% explained variance') # horizontal
plt.title("cumulative explained variance")
plt.xlabel("dimensions")
plt.ylabel("explained variance")
plt.legend()
plt.grid(True)
plt.show()


#%%  Salva X_pca in un dataframe aggiungendo le label

X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
X_pca_df['Label'] = df_clean['Label'].values

temp_df = df_clean[['Sample_ID', 'Map_ID', 'x', 'y']].reset_index(drop=True)
X_pca_df = pd.concat([temp_df, X_pca_df], axis=1)
print(X_pca_df.head())


# X_pca_df.to_parquet('X_pca_dataset.parquet')                    # save as parquet

#%%  Salva Vt come dataset

Vt_df = pd.DataFrame((Vt))
print(Vt_df)

# Vt_df.to_parquet('Vt_dataset.parquet') 

#%%  stampa lo spettro nello spazio PCA, eventualmente a dimensioni ridotte

n_relevant_PCs = 40

PC_columns = [f'PC{i+1}' for i in range(n_relevant_PCs)]
labels_clean = X_pca_df['Label'].unique()           # serie contenente tutte le labels di df
avg_spectra_clean = {}                              # dict per associare le labels a serie di spettri medi e std

for label in labels_clean:                          # riempio il dict avg_spectra
    temp_avg = X_pca_df.loc[X_pca_df['Label'] == label , PC_columns].mean()   
    temp_std = X_pca_df.loc[X_pca_df['Label'] == label , PC_columns].std()
    avg_spectra_clean[label] = (temp_avg, temp_std)

    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg)
    plt.fill_between(range(1,n_relevant_PCs + 1), temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"Spectrum in PC {n_relevant_PCs} dimensional space for label {label}")
    plt.xlabel("Band Index")
    plt.ylabel("Intensity")
    plt.grid(False)
    plt.show()
    

#%%  stampa lo spettro a dimensioni ridotte nello spazio k

n_relevant_PCs = 10

Vt_reduced = Vt[:n_relevant_PCs, :]                 # salvo la Vt a dim ridotte
y = df_clean['Label'].values                        # salvo la serie di labels in y

for label in labels_clean:
    mask = (y == label)                             # boolean string che mi dice quali righe hanno la label desiderata
    
    X_rec_label = X_pca[mask, :n_relevant_PCs] @ Vt_reduced + pca.mean_     # ricostruisco solo le righe di tale label
    
    temp_avg = X_rec_label.mean(axis=0)
    temp_std = X_rec_label.std(axis=0)
    
    orig_avg = avg_spectra[label][0].values         # serie delle avg originali
    
    plt.figure(figsize=(10, 5))
    plt.plot(temp_avg, label='Reconstructed')
    plt.plot(orig_avg, label='Original', linestyle='--')
    plt.fill_between(range(len(band_columns)),
                     temp_avg - temp_std, temp_avg + temp_std, alpha=0.2)
    plt.title(f"Spectrum for label {label} (k={n_relevant_PCs})")
    plt.xlabel("Band Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(False)
    plt.show()
    
    del X_rec_label
    gc.collect()
    
    



