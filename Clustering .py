#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as p
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[105]:


#load data
df = pd.read_csv('viruses.txt', sep='\t', header=0, dtype=str, on_bad_lines='skip')

#Save to csv
df.to_csv('viruses.csv', index=False)


# In[106]:


print(df.columns)
print(df.head())


# In[107]:


df.rename(columns={
    '#Organism/Name': 'organism_name',
    'TaxID': 'taxid',
    'BioProject Accession': 'bioproject_accession',
    'BioProject ID': 'bioproject_id',
    'Group': 'group',
    'SubGroup': 'subgroup',
    'Size (Kb)': 'size_kb',
    'GC%': 'gc_content',
    'Host': 'host',
    'Segmemts': 'segments',  # Correcting misspelling here
    'Genes': 'genes',
    'Proteins': 'proteins',
    'Release Date': 'release_date',
    'Modify Date': 'modify_date',
    'Status': 'status'
}, inplace=True)

print(df.columns)


# In[108]:


print(df.isnull().sum())


# In[109]:


columns_to_drop = ['bioproject_accession', 'bioproject_id', 
                   'release_date', 'modify_date', 'status', 'taxid']

df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')


# In[110]:


cols_numeric = ['size_kb', 'gc_content', 'genes', 'proteins']

for col in cols_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[111]:


df['group'] = df['group'].astype('category')
df['subgroup'] = df['subgroup'].astype('category')
df['host'] = df['host'].astype('category')
df['segments'] = df['segments'].astype('category')


# In[112]:


print(df.isnull().sum())


# In[113]:


df.dropna(subset=cols_numeric, inplace=True)


# In[114]:


feature_cols = ['size_kb', 'gc_content', 'genes', 'proteins']
X = df[feature_cols].copy()


# In[115]:


len(X)


# In[116]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  


# In[117]:


X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

#reference info 
reference_cols = ['organism_name'] 
df_final = pd.concat([df[reference_cols].reset_index(drop=True), X_scaled_df], axis=1)


df_final.to_csv('viruses_preprocessed.csv', index=False)

print(df_final.head())
print("Preprocessing complete. 'viruses_preprocessed.csv' created.")


# In[118]:


#execute kmeans

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)


# In[119]:


#execute gaussian mixture model 

gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)


# In[120]:


#silhouette scores

kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
gmm_sil = silhouette_score(X_scaled, gmm_labels)

print("KMeans Silhouette Score:", kmeans_sil)
print("GMM Silhouette Score:", gmm_sil)


# In[121]:


#vary numbers of clusters

ks = range(2, 10)
kmeans_sils = []
gmm_sils = []

for k in ks:
    # KMeans
    km = KMeans(n_clusters=k, random_state=42)
    km_labels = km.fit_predict(X_scaled)
    kmeans_sils.append(silhouette_score(X_scaled, km_labels))
    
    # GMM
    gm = GaussianMixture(n_components=k, random_state=42)
    gm_labels = gm.fit_predict(X_scaled)
    gmm_sils.append(silhouette_score(X_scaled, gm_labels))

plt.plot(ks, kmeans_sils, marker='o', label='KMeans Silhouette')
plt.plot(ks, gmm_sils, marker='x', label='GMM Silhouette')
plt.xlabel('Number of Clusters/Components')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()


# In[122]:


#dimensionality reduction

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
 
#plot kmeans results
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                       c=kmeans_labels, 
                       cmap='viridis', 
                       alpha=0.7, 
                       edgecolor='k')
ax1.set_title('KMeans Clusters (PCA 2D)', fontsize=12)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
cbar1 = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Cluster Label')

#plot GMM results
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                       c=gmm_labels, 
                       cmap='viridis', 
                       alpha=0.7, 
                       edgecolor='k')
ax2.set_title('GMM Clusters (PCA 2D)', fontsize=12)
ax2.set_xlabel('PC1')
cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Cluster Label')

plt.suptitle('Comparison: KMeans vs GMM Clusters in PCA Space', fontsize=14)
plt.tight_layout()
plt.show()


# In[123]:


from sklearn.metrics.cluster import adjusted_rand_score

df['kmeans_label'] = kmeans_labels
df['gmm_label'] = gmm_labels

ari_kmeans = adjusted_rand_score(df['group'], df['kmeans_label'])
ari_gmm = adjusted_rand_score(df['group'], df['gmm_label'])

print("ARI (KMeans vs. Group):", ari_kmeans)
print("ARI (GMM vs. Group):", ari_gmm)


# In[124]:


#count how many times "Other" appears in the 'group' column
other_count = (df['group'] == 'Other').sum()

#count how many rows are NOT "Other"
non_other_count = len(df) - other_count

print(f"Number of rows labeled 'Other': {other_count}")
print(f"Number of rows labeled with something else: {non_other_count}")

#all unique values in the 'group' column
print("\nValue counts for all groups:")
print(df['group'].value_counts())


# In[125]:


#all unique values in the 'group' column
print("\nValue counts for all groups:")
print(df['subgroup'].value_counts())


# In[126]:


from sklearn.metrics.cluster import adjusted_rand_score

df['kmeans_label'] = kmeans_labels
df['gmm_label'] = gmm_labels

ari_kmeans = adjusted_rand_score(df['subgroup'], df['kmeans_label'])
ari_gmm = adjusted_rand_score(df['subgroup'], df['gmm_label'])

print("ARI (KMeans vs. Subgroup):", ari_kmeans)
print("ARI (GMM vs. Subgroup):", ari_gmm)


# In[ ]:




