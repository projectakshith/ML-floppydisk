# Unit 3: C-MAPSS clustering and evaluation
# Extracted from Appendix A of Team_FloppyDisk_Extended_Final_Report.docx.
# For the original Colab notebook where available, see the matching .ipynb file.

import os
# Choose one: "cmapss", "custom_csv", or "synthetic_demo"
DATA_MODE = "cmapss"

# C-MAPSS subset. FD001 is a compact starting point.
CMAPSS_SUBSET = "FD001"
CMAPSS_DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/Team FloppyDisk Unit 3 - C-MAPSS"
LOCAL_CMAPSS_DATA_DIR = "/Users/akhilan_m/Downloads/CMAPSSData"
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass
if not os.path.isdir(CMAPSS_DATA_DIR) and os.path.isdir(LOCAL_CMAPSS_DATA_DIR):
    CMAPSS_DATA_DIR = LOCAL_CMAPSS_DATA_DIR

# Window size and stride are measured in engine cycles / rows.
WINDOW_SIZE = 20
WINDOW_STEP = 10
RANDOM_STATE = 42
MAX_CLUSTERS = 6

# Set this to a local Colab path if using custom_csv.
CUSTOM_CSV_PATH = "/content/engine_sensor_data.csv"

print("Mode:", DATA_MODE)
print("Window size:", WINDOW_SIZE, "cycles; step:", WINDOW_STEP)

if not os.path.isfile(os.path.join(CMAPSS_DATA_DIR, 'train_FD001.txt')):
    try:
        from google.colab import files
        print('Upload train_FD001.txt from the extracted C-MAPSS dataset.')
        files.upload()
        CMAPSS_DATA_DIR = os.getcwd()
    except ImportError:
        raise FileNotFoundError('Set CMAPSS_DATA_DIR to the folder containing train_FD001.txt')
import io
import os
import zipfile
from urllib.request import urlopen
import numpy as np
import pandas as pd


def make_synthetic_demo(seed=42, n_engines=18, n_cycles=100):
    """Create illustrative data to test code only. These are not measured engines."""
    rng = np.random.default_rng(seed)
    rows = []
    # Three deliberately simulated drift shapes, used only to exercise the pipeline.
    rates = np.repeat([0.002, 0.007, 0.014], n_engines // 3)
    rates = np.pad(rates, (0, n_engines - len(rates)), constant_values=0.007)
    rng.shuffle(rates)
    for e, rate in enumerate(rates, start=1):
        offset = rng.normal(0, 0.4)
        for cycle in range(1, n_cycles + 1):
            wear = cycle / n_cycles
            rows.append({
                "engine_id": e,
                "cycle": cycle,
                "temperature": 600 + offset + 8 * wear + rng.normal(0, 0.8),
                "pressure": 40 - 2.5 * wear + rng.normal(0, 0.3),
                "vibration": 1.0 + rate * cycle + rng.normal(0, 0.08),
            })
    return pd.DataFrame(rows)


def load_cmapss(subset="FD001", data_dir=CMAPSS_DATA_DIR):
    """Read NASA C-MAPSS training trajectories from an extracted local folder."""
    names = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    names += [f"sensor_{i}" for i in range(1, 22)]
    path = f"{data_dir}/train_{subset}.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found. Set data_dir to the extracted C-MAPSS folder.")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names)
    # RUL is kept only for post-fit validation. It is never used as a model feature.
    df["remaining_cycles"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    return df

def load_custom_csv(path=CUSTOM_CSV_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"engine_id", "cycle", "temperature", "pressure", "vibration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Custom CSV is missing required columns: {sorted(missing)}")
    return df

if DATA_MODE == "cmapss":
    df = load_cmapss(CMAPSS_SUBSET, CMAPSS_DATA_DIR)
    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
    synthetic_demo = False
elif DATA_MODE == "custom_csv":
    df = load_custom_csv(CUSTOM_CSV_PATH)
    excluded = {"engine_id", "cycle", "remaining_cycles"}
    sensor_columns = [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]
    synthetic_demo = False
elif DATA_MODE == "synthetic_demo":
    df = make_synthetic_demo(seed=RANDOM_STATE)
    sensor_columns = ["temperature", "pressure", "vibration"]
    synthetic_demo = True
else:
    raise ValueError("DATA_MODE must be cmapss, custom_csv, or synthetic_demo")

if len(sensor_columns) < 2:
    raise ValueError("At least two numeric sensor columns are required")

print("Rows:", len(df), "| Engines:", df["engine_id"].nunique())
print("Sensor channels used:", len(sensor_columns))
display(df.head())

def make_window_features(data, sensors, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    feature_rows, metadata_rows = [], []
    data = data.sort_values(["engine_id", "cycle"])
    for engine_id, group in data.groupby("engine_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, step):
            window = group.iloc[start:start + window_size]
            feature = {}
            for col in sensors:
                values = window[col].astype(float).to_numpy()
                feature[f"{col}_mean"] = float(np.mean(values))
                feature[f"{col}_std"] = float(np.std(values, ddof=0))
                feature[f"{col}_slope"] = float(np.polyfit(np.arange(len(values)), values, 1)[0])
            feature_rows.append(feature)
            meta = {
                "engine_id": engine_id,
                "start_cycle": window["cycle"].iloc[0],
                "end_cycle": window["cycle"].iloc[-1],
            }
            if "remaining_cycles" in window.columns:
                meta["remaining_cycles"] = float(window["remaining_cycles"].iloc[-1])
            metadata_rows.append(meta)
    X = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(metadata_rows)
    valid = X.notna().all(axis=1)
    return X.loc[valid].reset_index(drop=True), meta.loc[valid].reset_index(drop=True)

X, meta = make_window_features(df, sensor_columns)
if len(X) < 4 or meta["engine_id"].nunique() < 3:
    raise ValueError("Not enough complete windows or engines. Reduce WINDOW_SIZE or check the dataset.")
print("Window rows:", len(X), "| engineered features:", X.shape[1])
display(pd.concat([meta.head(), X.head()], axis=1))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, groups=meta["engine_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
meta_train, meta_test = meta.iloc[train_idx].reset_index(drop=True), meta.iloc[test_idx].reset_index(drop=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

max_k = min(MAX_CLUSTERS, len(X_train) - 1)
if max_k < 2:
    raise ValueError("Training split is too small for cluster comparison")

score_rows = []
for k in range(2, max_k + 1):
    candidate = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
    labels = candidate.fit_predict(X_train_scaled)
    score_rows.append({
        "k": k,
        "silhouette_higher_better": silhouette_score(X_train_scaled, labels),
        "davies_bouldin_lower_better": davies_bouldin_score(X_train_scaled, labels),
        "calinski_harabasz_higher_better": calinski_harabasz_score(X_train_scaled, labels),
        "inertia_lower_better_for_same_k": candidate.inertia_,
    })

scores = pd.DataFrame(score_rows).sort_values("silhouette_higher_better", ascending=False)
display(scores)
selected_k = int(scores.iloc[0]["k"])
print("Selected k by training silhouette only:", selected_k)
print("Treat this as a candidate. Review cluster usefulness and stability before choosing a maintenance policy.")

kmeans = KMeans(n_clusters=selected_k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
kmeans.fit(X_train_scaled)
train_labels = kmeans.predict(X_train_scaled)
test_labels = kmeans.predict(X_test_scaled)

train_assigned = meta_train.copy()
train_assigned["cluster"] = train_labels
train_assigned["split"] = "train"
test_assigned = meta_test.copy()
test_assigned["cluster"] = test_labels
test_assigned["split"] = "held-out engines"
assigned = pd.concat([train_assigned, test_assigned], ignore_index=True)

test_silhouette = np.nan
if len(np.unique(test_labels)) > 1 and len(np.unique(test_labels)) < len(test_labels):
    test_silhouette = silhouette_score(X_test_scaled, test_labels)
print("Held-out engine windows:", len(test_assigned))
print("Held-out silhouette:", round(test_silhouette, 3) if np.isfinite(test_silhouette) else "not defined for this assignment")
display(assigned.head(10))

import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

km_eval_labels = kmeans.predict(X_train_scaled)
ward_eval_labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(X_train_scaled)
comparison_metrics = pd.DataFrame([
    {"method": "K-Means", "silhouette": silhouette_score(X_train_scaled, km_eval_labels), "davies_bouldin": davies_bouldin_score(X_train_scaled, km_eval_labels), "calinski_harabasz": calinski_harabasz_score(X_train_scaled, km_eval_labels)},
    {"method": "Ward Agglomerative", "silhouette": silhouette_score(X_train_scaled, ward_eval_labels), "davies_bouldin": davies_bouldin_score(X_train_scaled, ward_eval_labels), "calinski_harabasz": calinski_harabasz_score(X_train_scaled, ward_eval_labels)},
])
print("Cluster quality (training windows)")
print(comparison_metrics.round(3).to_string(index=False))
print("K-Means held-out silhouette:", round(float(test_silhouette), 3))

training_engine_ids = np.array(sorted(meta_train["engine_id"].unique()))
sample_engine_count = max(2, int(np.ceil(0.8 * len(training_engine_ids))))
rng = np.random.default_rng(RANDOM_STATE)
subsample_labels = {"K-Means": [], "Ward Agglomerative": []}
for repetition in range(20):
    sampled_engines = rng.choice(training_engine_ids, size=sample_engine_count, replace=False)
    sample_idx = np.flatnonzero(meta_train["engine_id"].isin(sampled_engines).to_numpy())
    sample_features = X_train_scaled[sample_idx]
    sample_engine_by_window = meta_train.iloc[sample_idx]["engine_id"].to_numpy()
    km_sample = KMeans(n_clusters=selected_k, init="k-means++", n_init=20, random_state=RANDOM_STATE + repetition).fit_predict(sample_features)
    ward_sample = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(sample_features)
    subsample_labels["K-Means"].append((sample_engine_by_window, km_sample))
    subsample_labels["Ward Agglomerative"].append((sample_engine_by_window, ward_sample))

stability_rows = []
for method, runs in subsample_labels.items():
    pairwise_ari = []
    for first, second in combinations(range(len(runs)), 2):
        ids_first, labels_first = runs[first]
        ids_second, labels_second = runs[second]
        shared_engines = np.intersect1d(ids_first, ids_second)
        first_mask, second_mask = np.isin(ids_first, shared_engines), np.isin(ids_second, shared_engines)
        pairwise_ari.append(adjusted_rand_score(labels_first[first_mask], labels_second[second_mask]))
    stability_rows.append({"method": method, "mean_pairwise_ari": float(np.mean(pairwise_ari)), "sd_pairwise_ari": float(np.std(pairwise_ari, ddof=1)), "subsample_pairs": len(pairwise_ari)})
stability_summary = pd.DataFrame(stability_rows)
print("\nStability (higher mean pairwise ARI indicates greater agreement)")
print(stability_summary.round(3).to_string(index=False))
ax=stability_summary.set_index("method")["mean_pairwise_ari"].plot(kind="bar", color=["#2459A6", "#77818C"], figsize=(6,3), legend=False)
ax.set_ylim(0,1); ax.set_ylabel("Mean pairwise adjusted Rand index"); ax.set_xlabel(""); ax.set_title("FD001 training-window subsample stability")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

ordered = scores.sort_values('k')
plt.figure(figsize=(8,4))
plt.plot(ordered.k, ordered.inertia_lower_better_for_same_k, 'o-')
plt.xlabel('Number of clusters K'); plt.ylabel('Within-cluster squared distance (inertia)')
plt.title('Elbow plot — FD001 training windows'); plt.xticks(ordered.k); plt.grid(alpha=.25); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(ordered.k, ordered.silhouette_higher_better, 'o-')
plt.axvline(selected_k, color='orange', linestyle='--', label=f'Selected K={selected_k}')
plt.xlabel('Number of clusters K'); plt.ylabel('Mean silhouette'); plt.title('Cluster-count comparison'); plt.xticks(ordered.k); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.show()
print('Selected K:', selected_k)

from sklearn.decomposition import PCA
pca = PCA().fit(X_train_scaled)
ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(ratio)
components = np.arange(1, len(ratio)+1)
n90 = int(np.searchsorted(cumulative,.90)+1)
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(components,ratio,'o-',markersize=3); axes[0].set(title='PCA scree plot',xlabel='Component',ylabel='Explained variance fraction')
axes[1].plot(components,cumulative); axes[1].axhline(.9,color='orange',linestyle='--'); axes[1].axvline(n90,color='grey',linestyle=':'); axes[1].set(title=f'{n90} components retain 90% variance',xlabel='Number of components',ylabel='Cumulative explained variance',ylim=(0,1.02))
plt.tight_layout(); plt.show()

projection=pca.transform(X_train_scaled)[:,:2]
centres=pca.transform(kmeans.cluster_centers_)[:,:2]
plt.figure(figsize=(9,6))
for label in range(selected_k):
    mask=train_labels==label
    plt.scatter(projection[mask,0],projection[mask,1],s=12,alpha=.45,label=f'Cluster {label}')
plt.scatter(centres[:,0],centres[:,1],marker='X',s=180,c='black',label='Projected centres')
plt.xlabel(f'PC1 ({ratio[0]:.1%} variance)'); plt.ylabel(f'PC2 ({ratio[1]:.1%} variance)'); plt.title('K-Means clusters in a two-dimensional PCA view'); plt.legend(); plt.tight_layout(); plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram
ward_tree=linkage(X_train_scaled, method='ward')
plt.figure(figsize=(12,5))
dendrogram(ward_tree,truncate_mode='lastp',p=25,show_leaf_counts=True)
plt.xlabel('Merged groups (numbers in parentheses are window counts)'); plt.ylabel('Ward linkage distance'); plt.title('Ward hierarchical clustering — truncated dendrogram'); plt.tight_layout(); plt.show()

fig,ax=plt.subplots(figsize=(16,4))
limit=max(abs(kmeans.cluster_centers_).max(),.01)
im=ax.imshow(kmeans.cluster_centers_,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
ax.set_xticks(range(len(X.columns))); ax.set_xticklabels(X.columns,rotation=90,fontsize=7)
ax.set_yticks(range(selected_k)); ax.set_yticklabels([f'Cluster {i}' for i in range(selected_k)])
ax.set_title('K-Means centres — all 63 standardized sensor features'); fig.colorbar(im,ax=ax,label='Standard deviations from training mean'); plt.tight_layout(); plt.show()

print('Metadata columns:', list(meta_train.columns))
rul_col=next(c for c in meta_train.columns if 'remaining' in c.lower() or 'rul' in c.lower())
rul_groups=[meta_train.loc[train_labels==i,rul_col].dropna().to_numpy() for i in range(selected_k)]
plt.figure(figsize=(8,5)); plt.boxplot(rul_groups,tick_labels=[f'Cluster {i}' for i in range(selected_k)],showfliers=False)
plt.ylabel('Cycles remaining until simulated failure'); plt.title('Training-window remaining life by K-Means cluster'); plt.tight_layout(); plt.show()
for i,values in enumerate(rul_groups): print(f'Cluster {i}: {len(values)} windows; median remaining life {np.median(values):.0f} cycles')
