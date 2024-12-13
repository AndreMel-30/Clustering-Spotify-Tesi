# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1qJeLegPx6jZzCFHgz6Wubil6fH_r0hTM
"""

pip install hdbscan

# carico il file JSON di Kaggle (contiene l'API)
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
#import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle   # Creo la cartella nascosta .kaggle nella home se non esiste già
!cp kaggle.json ~/.kaggle/  # Copio il file kaggle.json appena caricato nella cartella .kaggle
!chmod 600 ~/.kaggle/kaggle.json  # Imposto i permessi di lettura e scrittura solo per l’utente

!kaggle datasets download -d maharshipandya/-spotify-tracks-dataset # download del dataset

# Estraggo il file ZIP scaricato in una cartella che chiamo dataset
with zipfile.ZipFile("-spotify-tracks-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# Carica il file CSV in un DataFrame Pandas
df = pd.read_csv('dataset/dataset.csv')
df = df.drop_duplicates() # rimuove eventuali duplicati (le righe duplicate), utile nel datacleaning
df.shape
df.isnull().sum()
colonne_da_eliminare = [ 'track_name','album_name','track_name', 'artists','key','mode','time_signature','explicit', 'speechiness','Unnamed: 0', "popularity"]
df = df.drop(columns=colonne_da_eliminare)
df.to_csv('nuovo_dataset.csv', index=False)
# feature scaling


# Selezione colonne numeriche da scalare
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# istanza della classe MinMaxScaler
scaler = MinMaxScaler()
# calcolo il valore di minino e massimo di ogni colonna
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df.info())

# Crea una copia del dataset originale per non modificarlo direttamente
df_cleaned = df.copy()

# Loop su tutte le colonne per calcolare i limiti IQR
numeric_cols = df_cleaned.select_dtypes(include=['float64']).columns

# Crea un filtro basato sulle colonne numeriche
for col in numeric_cols:
    # primo quartile
    Q1 = df_cleaned[col].quantile(0.25)
    # terzo quartile
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calcolo dei limiti
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    # Filtra il dataset per rimuovere gli outlier
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

print(f"Righe originali: {df.shape[0]}")
print(f"Righe dopo la pulizia: {df_cleaned.shape[0]}")
print(df_cleaned.info())

# Riduco il dataset a 5 componenti usando PCA
pca = PCA(n_components=5, random_state=42)
df_pca = pca.fit_transform(df_cleaned)

# Considero un campione del 20% dei dati per ridurre il tempo di esecuzione di t-SNE
df_pca_sampled = df_pca[:int(0.2 * len(df_pca))]

# Applico t-SNE sul campione ridotto da PCA
tsne = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
df_tsne = tsne.fit_transform(df_pca_sampled)

# Converto in DataFrame per una visualizzazione
df_tsne = pd.DataFrame(df_tsne, columns=['TSNE1', 'TSNE2'])

# Visualizza il risultato
plt.figure(figsize=(10, 8))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], alpha=0.5)
plt.xlabel("t-SNE Componente 1")
plt.ylabel("t-SNE Componente 2")
plt.title("Visualizzazione del dataset tramite PCA + t-SNE")
plt.show()

# Funzione per valutare i cluster
def valuta_clustering(data, labels):
    if len(set(labels)) > 1:  # Deve esserci più di un cluster valido
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        return silhouette, davies_bouldin
    else:
        print("Unico cluster trovato o solo rumore.")
        return None, None

# Funzione per determinare il numero ottimale di k utilizzando la regola del
# gomito (Elbow Rule)
def find_optimal_k(df_scaled):
    inerzia = []
    i = range(1, 11)
    # calcoliamo l'inerzia per diversi valori di k
    for k in i:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df_scaled)
        inerzia.append(kmeans.inertia_)


    plt.plot(i, inerzia, 'bx-')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Inerzia')
    plt.title('Elbow Method per selezionare il k ottimale')
    plt.grid()
    plt.show()

find_optimal_k(df_scaled=df_cleaned)

silhouette = [0.,0.267,0.325,0.259,0.232,0.227]
n_cluster = range(2,8)
plt.plot(n_cluster, silhouette, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('silhouette score')
plt.title('Selecting the number of clusters k using the silhouette score')
plt.grid()
plt.show()

# Applica k-means
df_no_track_id = df_cleaned.drop(columns=['track_id',"track_genre"])
print(df_no_track_id.info())
kmeans = KMeans(n_clusters=4, random_state=42)
#  vettore che contiene le etichette di cluster assegnate a ciascun punto del tuo dataset da K-Means
kmeans_labels = kmeans.fit_predict(df_no_track_id)

# Valuta k-means
print("k-means:")
valuta_clustering(df_no_track_id, kmeans_labels)

# Riduzione dimensionale a 2D
pca = PCA(n_components=6)
data_2d = pca.fit_transform(df_no_track_id)

# Visualizzazione cluster
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title("Visualizzazione dei cluster con K-means (PCA)")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # Per i grafici 3D

# Applica k-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(df_no_track_id)

# Valuta k-means
print("k-means:")
valuta_clustering(df_no_track_id, kmeans_labels)

# Riduzione dimensionale a 3D con PCA
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned)

# Visualizzazione cluster in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # Crea un plot 3D
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_labels, cmap='viridis', alpha=0.7)

# Aggiungi etichette e legenda
ax.set_title("Visualizzazione dei cluster con K-means (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")
plt.colorbar(scatter)
plt.show()

# Applica k-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(df_cleaned)

# Valuta k-means
print("k-means:")
valuta_clustering(df_cleaned, kmeans_labels)

# Riduzione dimensionale a 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned)

# Mappa dei colori per i cluster
cluster_colors = {0: "viola", 1: "blu", 2: "verde", 3: "giallo"}

# Visualizzazione cluster in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # Crea un plot 3D
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_labels, cmap='viridis', alpha=0.7)

# Aggiungi etichette e legenda
ax.set_title("Visualizzazione dei cluster con K-means (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")

# Creazione legenda personalizzata
legend_elements = []
for cluster_num, color_name in cluster_colors.items():
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_num} = {color_name}',
                                      markerfacecolor=plt.cm.viridis(cluster_num / 3), markersize=10))

# Posiziona la legenda accanto al grafico
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Mappa Cluster-Colore")

plt.show()

# Crea un DataFrame con i dati e i cluster
df_clustered = pd.DataFrame(df_no_track_id, columns=['duration_ms', 'danceability','energy','loudness','acousticness','instrumentalness','liveness','valence','tempo'])
df_clustered['Cluster'] = kmeans_labels

# Seleziono i punti per ogni cluster
cluster_0 = df_clustered[df_clustered['Cluster'] == 0]
cluster_1 = df_clustered[df_clustered['Cluster'] == 1]
cluster_2 = df_clustered[df_clustered['Cluster'] == 2]
cluster_3 = df_clustered[df_clustered['Cluster'] == 3]

# estraggo le features per ogni cluster
for i in range(4):  # Supponendo k=4
    print(f"Statistiche per il cluster {i}:")
    print(df_clustered[df_clustered['Cluster'] == i].describe())
    print("\n")

print(df_cleaned.info())

# Calcolo la media delle features per ogni cluster
cluster_means = df_clustered.groupby('Cluster').mean()
colors = ['purple', 'blue', 'green', 'yellow']

# Plot delle medie delle features
cluster_means.T.plot(kind='bar', figsize=(12, 8), color=colors)
plt.title("Medie delle features per cluster")
plt.xlabel("Features")
plt.ylabel("Media")
plt.legend(title="Cluster")
plt.show()

# Copiamo il dataset originale per includere i cluster
df_cleaned['Cluster'] = df_clustered['Cluster']

# Analisi: distribuzione dei generi nei cluster
distribuzione = df_cleaned.groupby('Cluster')['track_genre'].value_counts(normalize=True) * 100

# Mostra la distribuzione dei generi per ogni cluster
print(distribuzione)

# Verifica la somma delle percentuali per ogni cluster
for cluster in df_cleaned['Cluster'].unique():
    distribuzione_cluster = df_cleaned[df_cleaned['Cluster'] == cluster]['track_genre'].value_counts(normalize=True) * 100
    somma_percentuali = distribuzione_cluster.sum()
    print(f"\nSomma delle percentuali per il cluster {cluster}: {somma_percentuali:.2f}%")
    print(distribuzione_cluster)

# Eseguiamo la distribuzione dei generi nei cluster
distribuzione = df_cleaned.groupby('Cluster')['track_genre'].value_counts(normalize=True) * 100

# Visualizza un grafico per il cluster 1
distribuzione_cluster_1 = distribuzione.loc[1]
distribuzione_cluster_1.plot(kind='bar', figsize=(10, 6))
plt.title("Distribuzione dei generi per il Cluster 1")
plt.xlabel("Genere musicale")
plt.ylabel("Percentuale")
plt.show()

silhouette = [0.319,0.303,0.325,0.259,0.232,0.227]
n_cluster = range(2,8)
plt.plot(n_cluster, silhouette, 'bx-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('silhouette score')
plt.title('Silhouette score al variare del numero di cluster (k)')
plt.grid()
plt.show()

"""partiamo con l'analisi tramite DBSCAN"""

print(df_cleaned.info())

# numero di vicini (minPts)
k = 5
minPts = k

# Calcolo le distanze k-dist per ogni punto nel dataset
neighbors = NearestNeighbors(n_neighbors=k) #  trovo i k-vicini più vicini di ogni punto nel dataset.
neighbors_fit = neighbors.fit(df_cleaned) # calcolo i vicini per ogni punto del dataset
distances = neighbors_fit.kneighbors(df_cleaned) #  che contiene le k-dist

# Ordino le distanze in ordine decrescente per identificare la soglia epsilon
distances = np.sort(distances[:, -1])[::-1]  # prendo l'ultima colonna e ordina in ordine decrescente
plt.plot(distances)
plt.title("Grafico delle k-dist per scegliere epsilon")
plt.xlabel("Punti ordinati")
plt.ylabel(f"Distanza del {k}-esimo vicino")
plt.show()

k = 5
minPts = k # numero di vicini (minPts)

# Calcolo le distanze k-dist per ogni punto nel dataset
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(df_cleaned)
distances = neighbors_fit.kneighbors(df_cleaned)

# Ordino le distanze in ordine decrescente per identificare la soglia epsilon
# prendo l'ultima colonna e ordina in ordine decrescente
distances = np.sort(distances[:, -1])[::-1]
plt.plot(distances)
plt.title("Grafico delle k-dist per scegliere epsilon")
plt.xlabel("Punti ordinati")
plt.ylabel(f"Distanza del {k}-esimo vicino")
plt.show()

# Applicazione DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_cleaned)

# Valuto DBSCAN (ignoro i punti rumorosi, ovvero etichettati come -1)
print("DBSCAN:")
if len(set(dbscan_labels)) > 1:
    valuta_clustering(df_cleaned[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    print("DBSCAN ha trovato solo rumore o un unico cluster.")

# Visualizzazione grafica

pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned)


cluster_colors = {0: "viola", 1: "blu", 2: "verde", 3: "giallo"}


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # Crea un plot 3D
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=dbscan_labels, cmap='viridis', alpha=0.7)


ax.set_title("Visualizzazione dei cluster con DBSCAN (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")

# Riduzione dimensionale a 2D
pca = PCA(n_components=6)
data_2d = pca.fit_transform(df_cleaned)

# Visualizzazione cluster
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title("Visualizzazione dei cluster con DBSCAN (PCA)")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.show()

# Prova un valore diverso di eps
eps_value = 0.15  # Prova un valore leggermente maggiore
dbscan = DBSCAN(eps=eps_value, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_cleaned)

# Controlla il numero di cluster trovati (escludendo il rumore)
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Numero di cluster trovati: {num_clusters}")

# Valuta i cluster (escludi il rumore per la silhouette score)
if num_clusters > 1:
    valuta_clustering(df_cleaned[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    print("DBSCAN ha trovato solo rumore o un unico cluster.")

# Visualizza i cluster e il rumore
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=dbscan_labels, cmap='viridis', alpha=0.5)
ax.set_title("Clustering con DBSCAN (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")
plt.colorbar(scatter)
plt.show()


hdbscan_clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=50)
hdbscan_labels = hdbscan_clusterer.fit_predict(df_cleaned)

# Visualizzazione clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2]  c=hdbscan_labels, cmap='viridis', alpha=0.5)
ax.set_title("Clustering con HDBSCAN")
plt.colorbar(scatter)
plt.show()
