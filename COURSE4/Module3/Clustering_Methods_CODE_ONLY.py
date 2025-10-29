# MODULE 3: CLUSTERING METHODS - CODE ONLY

# Import các thư viện cần thiết
import warnings
warnings.warn = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.cluster import hierarchy

# Thiết lập style
sns.set_context('notebook')
sns.set_style('white')

# Load dữ liệu Wine Quality
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/Wine_Quality_Data.csv")
print("Wine Quality dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head().T)

print(f"\nData types:\n{data.dtypes}")

# Thống kê về màu sắc và chất lượng
print(f"\nWine colors distribution:\n{data.color.value_counts()}")
print(f"\nQuality distribution:\n{data.quality.value_counts().sort_index()}")

# Visualization chất lượng theo màu sắc
red = sns.color_palette()[2]
white = sns.color_palette()[4]
bin_range = np.array([3, 4, 5, 6, 7, 8, 9])

ax = plt.axes()
for color, plot_color in zip(['red', 'white'], [red, white]):
    q_data = data.loc[data.color==color, 'quality']
    q_data.hist(bins=bin_range, 
                alpha=0.5, ax=ax, 
                color=plot_color, label=color)

ax.legend()
ax.set(xlabel='Quality', ylabel='Occurrence')
ax.set_xlim(3,10)
ax.set_xticks(bin_range+0.5)
ax.set_xticklabels(bin_range)
ax.grid('off')
plt.title('Quality Distribution by Wine Color')
plt.show()

# Chuẩn bị dữ liệu cho clustering
float_columns = [x for x in data.columns if x not in ['color', 'quality']]
print(f"\nFeatures for clustering: {float_columns}")

# Correlation matrix
corr_mat = data[float_columns].corr()
for x in range(len(float_columns)):
    corr_mat.iloc[x,x] = 0.0

print(f"\nCorrelation matrix:\n{corr_mat}")
print(f"\nMax correlations:\n{corr_mat.abs().idxmax()}")

# Kiểm tra skewness
skew_columns = (data[float_columns]
                .skew()
                .sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
print(f"\nSkewed columns (>0.75):\n{skew_columns}")

# Log transformation cho các cột có độ lệch cao
for col in skew_columns.index.tolist():
    data[col] = np.log1p(data[col])
    print(f"Applied log transformation to {col}")

# Feature scaling
sc = StandardScaler()
data[float_columns] = sc.fit_transform(data[float_columns])

print(f"\nAfter scaling - first 4 rows:")
print(data.head(4))

# Pairplot sau khi transform và scale
plt.figure(figsize=(12, 10))
sns.pairplot(data[float_columns + ['color']], 
             hue='color', 
             hue_order=['white', 'red'],
             palette={'red':red, 'white':'gray'})
plt.suptitle('Feature Relationships After Preprocessing', y=1.02)
plt.show()

# K-MEANS CLUSTERING
print("\n=== K-MEANS CLUSTERING ===")

# Fit K-means với 2 clusters
km = KMeans(n_clusters=2, random_state=42)
km = km.fit(data[float_columns])
data['kmeans'] = km.predict(data[float_columns])

# Phân tích kết quả clustering theo màu sắc
clustering_results = (data[['color','kmeans']]
 .groupby(['kmeans','color'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))

print("K-Means clustering results by wine color:")
print(clustering_results)

# Elbow Method để tìm số cluster tối ưu
km_list = list()

for clust in range(1,21):
    km = KMeans(n_clusters=clust, random_state=42)
    km = km.fit(data[float_columns])
    
    km_list.append(pd.Series({'clusters': clust, 
                              'inertia': km.inertia_,
                              'model': km}))

plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

plt.figure(figsize=(10, 6))
ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,21,2))
ax.set_xlim(0,21)
ax.set(xlabel='Number of Clusters', ylabel='Inertia')
ax.set_title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# AGGLOMERATIVE CLUSTERING
print("\n=== AGGLOMERATIVE CLUSTERING ===")

# Fit Agglomerative Clustering với 2 clusters
ag = AgglomerativeClustering(n_clusters=2, linkage='ward', compute_full_tree=True)
ag = ag.fit(data[float_columns])
data['agglom'] = ag.fit_predict(data[float_columns])

# So sánh kết quả Agglomerative và K-Means
print("Agglomerative clustering results:")
agglom_results = (data[['color','agglom']]
 .groupby(['agglom','color'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))
print(agglom_results)

print("\nK-Means clustering results:")
kmeans_results = (data[['color','kmeans']]
 .groupby(['kmeans','color'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))
print(kmeans_results)

# So sánh chi tiết
comparison = (data[['color','agglom','kmeans']]
 .groupby(['color','agglom','kmeans'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))
print("\nDetailed comparison:")
print(comparison)

# Dendrogram
Z = hierarchy.linkage(ag.children_, method='ward')

fig, ax = plt.subplots(figsize=(15,5))
den = hierarchy.dendrogram(Z, orientation='top', 
                           p=30, truncate_mode='lastp',
                           show_leaf_counts=True, ax=ax)
ax.set_title('Dendrogram for Agglomerative Clustering')
plt.show()

# CLUSTERING AS FEATURE ENGINEERING
print("\n=== CLUSTERING AS FEATURE ENGINEERING ===")

# Tạo binary target variable
y = (data['quality'] > 7).astype(int)
print(f"Target variable distribution:\n{y.value_counts()}")

# Tạo feature sets
X_with_kmeans = data.drop(['agglom', 'color', 'quality'], axis=1)
X_without_kmeans = X_with_kmeans.drop('kmeans', axis=1)

print(f"Features with K-means: {X_with_kmeans.shape}")
print(f"Features without K-means: {X_without_kmeans.shape}")

# Stratified Shuffle Split
sss = StratifiedShuffleSplit(n_splits=10, random_state=6532)

def get_avg_roc_10splits(estimator, X, y):
    """
    Tính ROC-AUC trung bình qua 10 splits
    """
    roc_auc_list = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)
        y_scored = estimator.predict_proba(X_test)[:, 1]
        
        roc_auc_list.append(roc_auc_score(y_test, y_scored))
    
    return np.mean(roc_auc_list)

# So sánh hiệu suất với và không có K-means cluster feature
estimator_rf = RandomForestClassifier()
roc_with_kmeans = get_avg_roc_10splits(estimator_rf, X_with_kmeans, y)
roc_without_kmeans = get_avg_roc_10splits(estimator_rf, X_without_kmeans, y)

print(f"Random Forest ROC-AUC without K-means: {roc_without_kmeans:.4f}")
print(f"Random Forest ROC-AUC with K-means: {roc_with_kmeans:.4f}")
print(f"Improvement: {roc_with_kmeans - roc_without_kmeans:.4f}")

# Khám phá ảnh hưởng của số lượng clusters
X_basis = data[float_columns]
sss = StratifiedShuffleSplit(n_splits=10, random_state=6532)

def create_kmeans_columns(n):
    """
    Tạo features từ KMeans clustering với n clusters
    """
    km = KMeans(n_clusters=n)
    km.fit(X_basis)
    km_col = pd.Series(km.predict(X_basis))
    km_cols = pd.get_dummies(km_col, prefix='kmeans_cluster')
    return pd.concat([X_basis, km_cols], axis=1)

estimator_lr = LogisticRegression()
ns = range(1, 21)
roc_auc_list = [get_avg_roc_10splits(estimator_lr, create_kmeans_columns(n), y)
                for n in ns]

plt.figure(figsize=(12, 6))
ax = plt.axes()
ax.plot(ns, roc_auc_list, marker='o')
ax.set(
    xticklabels= ns,
    xlabel='Number of clusters as features',
    ylabel='Average ROC-AUC over 10 iterations',
    title='KMeans + LogisticRegression Performance'
)
ax.grid(True)
plt.show()

# Tìm số cluster tối ưu
optimal_n = ns[np.argmax(roc_auc_list)]
max_roc_auc = max(roc_auc_list)

print(f"\nOptimal number of clusters: {optimal_n}")
print(f"Maximum ROC-AUC: {max_roc_auc:.4f}")

# Phân tích chi tiết kết quả clustering
print("\n=== PHÂN TÍCH CHI TIẾT ===")

# Phân tích cluster characteristics
cluster_analysis = data.groupby('kmeans')[float_columns].mean()
print("Cluster characteristics (mean values):")
print(cluster_analysis.round(3))

# Phân tích chất lượng theo cluster
quality_by_cluster = data.groupby('kmeans')['quality'].agg(['mean', 'std', 'count'])
print(f"\nQuality statistics by cluster:")
print(quality_by_cluster.round(3))

# Visualization cluster characteristics
plt.figure(figsize=(15, 10))

# Heatmap của cluster characteristics
plt.subplot(2, 2, 1)
sns.heatmap(cluster_analysis.T, annot=True, cmap='viridis', fmt='.2f')
plt.title('Cluster Characteristics Heatmap')

# Quality distribution by cluster
plt.subplot(2, 2, 2)
for cluster in data['kmeans'].unique():
    cluster_data = data[data['kmeans'] == cluster]['quality']
    plt.hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', bins=range(3, 10))
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Quality Distribution by Cluster')
plt.legend()

# Wine color distribution by cluster
plt.subplot(2, 2, 3)
color_cluster = pd.crosstab(data['kmeans'], data['color'])
color_cluster.plot(kind='bar', stacked=True)
plt.title('Wine Color Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Color')

# ROC-AUC vs Number of Clusters
plt.subplot(2, 2, 4)
plt.plot(ns, roc_auc_list, marker='o', linewidth=2, markersize=6)
plt.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_n}')
plt.xlabel('Number of Clusters')
plt.ylabel('ROC-AUC')
plt.title('Performance vs Number of Clusters')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Tổng kết
print("\n=== TỔNG KẾT ===")
print("Đã hoàn thành phân tích clustering methods:")
print("1. K-Means Clustering - Phân cụm dựa trên khoảng cách")
print("2. Agglomerative Clustering - Phân cụm phân cấp")
print("3. Elbow Method - Tìm số cluster tối ưu")
print("4. Dendrogram - Visualization phân cụm phân cấp")
print("5. Clustering as Feature Engineering - Sử dụng cluster làm feature")
print("6. Performance Analysis - So sánh hiệu suất với/sử dụng cluster features")
print(f"\nKết quả chính:")
print(f"- K-means cluster feature cải thiện ROC-AUC: {roc_with_kmeans - roc_without_kmeans:.4f}")
print(f"- Số cluster tối ưu cho Logistic Regression: {optimal_n}")
print(f"- ROC-AUC tối đa đạt được: {max_roc_auc:.4f}")
