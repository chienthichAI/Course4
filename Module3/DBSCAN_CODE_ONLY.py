# MODULE 3: DBSCAN CLUSTERING - CODE ONLY

# Import các thư viện cần thiết
import warnings
warnings.warn = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Thiết lập style
sns.set_context('notebook')
sns.set_style('white')

# Load dữ liệu synthetic clustering
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/synthetic_clustering.csv')
print("Synthetic clustering dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Visualization dữ liệu gốc
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Synthetic Clustering Data')
plt.grid(True, alpha=0.3)
plt.show()

# Thống kê cơ bản
print(f"\nData statistics:")
print(df.describe())

# DBSCAN CLUSTERING
print("\n=== DBSCAN CLUSTERING ===")

# Khởi tạo DBSCAN với các tham số
eps = 2
min_samples = 10
cluster = DBSCAN(eps=eps, min_samples=min_samples)

# Fit mô hình DBSCAN
cluster.fit(df)

# Tính số lượng clusters (không bao gồm noise có label = -1)
num_clusters = len(set(cluster.labels_) - {-1})
print(f"Số lượng clusters tìm được: {num_clusters}")
print(f"Các nhãn cluster: {sorted(set(cluster.labels_) - {-1})}")

# Tính % dữ liệu được đánh dấu là noise
noise_count = (cluster.labels_ == -1).sum()
total_count = len(cluster.labels_)
noise_percentage = 100 * noise_count / total_count

print(f"Số điểm noise: {noise_count}")
print(f"Tổng số điểm: {total_count}")
print(f"Phần trăm dữ liệu là noise: {noise_percentage:.2f}%")

# Visualization kết quả clustering
plt.figure(figsize=(12, 8))

# Lấy tất cả các nhãn unique từ kết quả clustering
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)

# Tạo colormap với số màu bằng số lượng nhãn
cmap = plt.cm.get_cmap('tab10', n_labels)

# Vẽ scatter plot cho từng cluster
for l in unique_labels:
    # Lọc dữ liệu theo nhãn cluster
    mask = cluster.labels_ == l
    
    # Vẽ scatter plot với màu và marker riêng cho từng cluster
    plt.scatter(
        df['x'][mask],  # Tọa độ x của cluster
        df['y'][mask],  # Tọa độ y của cluster
        c=[cmap(l)] if l >= 0 else ['Black'],  # Màu: theo colormap nếu là cluster, đen nếu là noise
        marker='ov'[l%2],  # Marker: 'o' hoặc 'v' xen kẽ giữa các cluster
        alpha=0.75,  # Độ trong suốt
        s=100,  # Kích thước điểm
        label=f'Cluster {l}' if l >= 0 else 'Noise')  # Nhãn legend

# Hiển thị legend bên ngoài biểu đồ
plt.legend(bbox_to_anchor=[1, 1])

# Thêm title và labels cho trục
plt.title('DBSCAN Clustering Visualization', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid(True, alpha=0.3)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Phân tích chi tiết kết quả
print("\n=== PHÂN TÍCH CHI TIẾT ===")

# Thống kê về từng cluster
cluster_stats = []
for label in sorted(unique_labels):
    if label == -1:  # Noise
        mask = cluster.labels_ == label
        count = mask.sum()
        cluster_stats.append({
            'Label': 'Noise',
            'Count': count,
            'Percentage': (count / total_count) * 100,
            'Mean_X': df['x'][mask].mean(),
            'Mean_Y': df['y'][mask].mean(),
            'Std_X': df['x'][mask].std(),
            'Std_Y': df['y'][mask].std()
        })
    else:  # Regular clusters
        mask = cluster.labels_ == label
        count = mask.sum()
        cluster_stats.append({
            'Label': f'Cluster {label}',
            'Count': count,
            'Percentage': (count / total_count) * 100,
            'Mean_X': df['x'][mask].mean(),
            'Mean_Y': df['y'][mask].mean(),
            'Std_X': df['x'][mask].std(),
            'Std_Y': df['y'][mask].std()
        })

cluster_df = pd.DataFrame(cluster_stats)
print("Cluster Statistics:")
print(cluster_df.round(3))

# Visualization cluster centers và noise
plt.figure(figsize=(10, 8))

# Vẽ tất cả điểm
plt.scatter(df['x'], df['y'], c=cluster.labels_, cmap='tab10', alpha=0.6, s=50)

# Vẽ cluster centers (trừ noise)
for label in sorted(unique_labels):
    if label != -1:  # Không vẽ center cho noise
        mask = cluster.labels_ == label
        center_x = df['x'][mask].mean()
        center_y = df['y'][mask].mean()
        plt.scatter(center_x, center_y, c='red', marker='x', s=200, linewidths=3, label=f'Center {label}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('DBSCAN Clustering with Centers')
plt.colorbar(label='Cluster Label')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Thử nghiệm với các tham số khác nhau
print("\n=== THỬ NGHIỆM VỚI CÁC THAM SỐ KHÁC NHAU ===")

# Test với các giá trị eps khác nhau
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15, 20]

plt.figure(figsize=(15, 10))

for i, eps in enumerate(eps_values):
    plt.subplot(2, 3, i+1)
    
    # Fit DBSCAN với eps khác nhau
    dbscan_test = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_test.fit(df)
    
    # Tính số clusters và noise
    n_clusters = len(set(dbscan_test.labels_) - {-1})
    n_noise = list(dbscan_test.labels_).count(-1)
    
    # Vẽ kết quả
    plt.scatter(df['x'], df['y'], c=dbscan_test.labels_, cmap='tab10', alpha=0.6)
    plt.title(f'eps={eps}, min_samples={min_samples}\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()
plt.show()

# Test với các giá trị min_samples khác nhau
plt.figure(figsize=(15, 10))

for i, min_samples in enumerate(min_samples_values):
    plt.subplot(2, 2, i+1)
    
    # Fit DBSCAN với min_samples khác nhau
    dbscan_test = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_test.fit(df)
    
    # Tính số clusters và noise
    n_clusters = len(set(dbscan_test.labels_) - {-1})
    n_noise = list(dbscan_test.labels_).count(-1)
    
    # Vẽ kết quả
    plt.scatter(df['x'], df['y'], c=dbscan_test.labels_, cmap='tab10', alpha=0.6)
    plt.title(f'eps={eps}, min_samples={min_samples}\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()
plt.show()

# K-Nearest Neighbors để tìm eps tối ưu
print("\n=== TÌM EPS TỐI ƯU BẰNG K-NEAREST NEIGHBORS ===")

# Tính khoảng cách đến k-nearest neighbors
k = min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(df)
distances, indices = nbrs.kneighbors(df)

# Lấy khoảng cách đến neighbor thứ k (index k-1)
k_distances = distances[:, k-1]
k_distances = np.sort(k_distances)[::-1]  # Sắp xếp giảm dần

# Vẽ k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances)
plt.xlabel('Points (sorted by k-distance)')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-Distance Graph for Optimal Eps Selection')
plt.grid(True, alpha=0.3)

# Đánh dấu một số giá trị eps tiềm năng
potential_eps = [0.5, 1.0, 1.5, 2.0]
for eps in potential_eps:
    plt.axhline(y=eps, color='red', linestyle='--', alpha=0.7, label=f'eps={eps}')

plt.legend()
plt.show()

# Phân tích hiệu suất với các tham số khác nhau
print("\n=== PHÂN TÍCH HIỆU SUẤT ===")

results = []
for eps in np.arange(0.5, 3.5, 0.5):
    for min_samples in range(5, 25, 5):
        dbscan_test = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_test.fit(df)
        
        n_clusters = len(set(dbscan_test.labels_) - {-1})
        n_noise = list(dbscan_test.labels_).count(-1)
        noise_ratio = n_noise / len(df)
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio
        })

results_df = pd.DataFrame(results)
print("Performance Analysis:")
print(results_df.head(10))

# Visualization hiệu suất
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Số clusters vs eps và min_samples
pivot_clusters = results_df.pivot(index='min_samples', columns='eps', values='n_clusters')
sns.heatmap(pivot_clusters, annot=True, fmt='d', ax=axes[0], cmap='viridis')
axes[0].set_title('Number of Clusters')

# Noise ratio vs eps và min_samples
pivot_noise = results_df.pivot(index='min_samples', columns='eps', values='noise_ratio')
sns.heatmap(pivot_noise, annot=True, fmt='.2f', ax=axes[1], cmap='Reds')
axes[1].set_title('Noise Ratio')

# Số noise points vs eps và min_samples
pivot_noise_count = results_df.pivot(index='min_samples', columns='eps', values='n_noise')
sns.heatmap(pivot_noise_count, annot=True, fmt='d', ax=axes[2], cmap='Oranges')
axes[2].set_title('Number of Noise Points')

plt.tight_layout()
plt.show()

# Tìm tham số tối ưu (ít noise, số cluster hợp lý)
optimal_params = results_df[
    (results_df['noise_ratio'] < 0.3) &  # Ít hơn 30% noise
    (results_df['n_clusters'] >= 2) &    # Ít nhất 2 clusters
    (results_df['n_clusters'] <= 10)     # Không quá 10 clusters
].sort_values('noise_ratio')

print(f"\nOptimal parameters (low noise, reasonable clusters):")
print(optimal_params.head())

if len(optimal_params) > 0:
    best_eps = optimal_params.iloc[0]['eps']
    best_min_samples = optimal_params.iloc[0]['min_samples']
    
    print(f"\nBest parameters:")
    print(f"eps: {best_eps}")
    print(f"min_samples: {best_min_samples}")
    
    # Fit với tham số tối ưu
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_dbscan.fit(df)
    
    # Visualization kết quả tốt nhất
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y'], c=best_dbscan.labels_, cmap='tab10', alpha=0.7, s=80)
    plt.title(f'Best DBSCAN Results (eps={best_eps}, min_samples={best_min_samples})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Cluster Label')
    plt.grid(True, alpha=0.3)
    plt.show()

# Tổng kết
print("\n=== TỔNG KẾT ===")
print("Đã hoàn thành phân tích DBSCAN clustering:")
print("1. DBSCAN với tham số cơ bản (eps=2, min_samples=10)")
print("2. Visualization kết quả clustering")
print("3. Phân tích noise và clusters")
print("4. Thử nghiệm với các tham số khác nhau")
print("5. K-Nearest Neighbors để tìm eps tối ưu")
print("6. Phân tích hiệu suất và tìm tham số tối ưu")
print(f"\nKết quả chính:")
print(f"- Số clusters tìm được: {num_clusters}")
print(f"- Phần trăm noise: {noise_percentage:.2f}%")
print(f"- DBSCAN hiệu quả trong việc tìm clusters có hình dạng bất kỳ")
print(f"- Tham số eps và min_samples ảnh hưởng lớn đến kết quả")
