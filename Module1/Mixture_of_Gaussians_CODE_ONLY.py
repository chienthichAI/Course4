# MODULE 1: GAUSSIAN MIXTURE MODELS - CODE ONLY

# Import các thư viện cần thiết
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from itertools import chain
from matplotlib.patches import Ellipse

sns.set_context('notebook')
sns.set_style('white')

# Hàm vẽ Gaussian mixture 1D
def plot_univariate_mixture(means, stds, weights, N = 10000, seed=10):
    np.random.seed(seed)
    if not len(means)==len(stds)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    X = np.fromiter((ss.norm.rvs(loc=means[i], scale=stds[i]) for i in mixture_idx), dtype=np.float64)
      
    xs = np.linspace(X.min(), X.max(), 300)
    ps = np.zeros_like(xs)
    
    for mu, s, w in zip(means, stds, weights):
        ps += ss.norm.pdf(xs, loc=mu, scale=s) * w
    
    fig, ax = plt.subplots()
    ax.plot(xs, ps, label='pdf of the Gaussian mixture')
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("P", fontsize=15)
    ax.set_title("Univariate Gaussian mixture", fontsize=15)
    
    return X.reshape(-1,1), fig, ax

# Hàm vẽ Gaussian mixture 2D
def plot_bivariate_mixture(means, covs, weights, N = 10000, seed=10):
    np.random.seed(seed)
    if not len(means)==len(covs)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    M = len(means[0])
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    
    X = np.fromiter(chain.from_iterable(multivariate_normal.rvs(mean=means[i], cov=covs[i]) for i in mixture_idx), 
                dtype=float)
    X.shape = N, M
    
    xs1 = X[:,0] 
    xs2 = X[:,1]
    
    plt.scatter(xs1, xs2, label="data")
    
    L = len(means)
    for l, pair in enumerate(means):
        plt.scatter(pair[0], pair[1], color='red')
        if l == L-1:
            break
    plt.scatter(pair[0], pair[1], color='red', label="mean")
    
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Scatter plot of the bivariate Gaussian mixture")
    plt.legend()
    plt.show()
    
    return X

# Hàm vẽ ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

# Hàm vẽ GMM
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# Ví dụ 1: Tạo dữ liệu Gaussian mixture 1D
X1, fig1, ax1 = plot_univariate_mixture(means=[2,5,8], stds=[0.2, 0.5, 0.8], weights=[0.3, 0.3, 0.4])

# Ví dụ 2: Tạo dữ liệu Gaussian mixture 1D với std lớn hơn
X2, fig2, ax2 = plot_univariate_mixture(means=[2,5,8], stds=[0.6, 0.9, 1.2], weights=[0.3, 0.3, 0.4])

# Ví dụ 3: Thay đổi weights
X3, fig3, ax3 = plot_univariate_mixture(means=[2,5,8], stds=[0.6, 0.9, 1.2], weights=[0.05, 0.35, 0.6])

# Áp dụng GMM và vẽ xác suất
X1_sorted = np.sort(X1.reshape(-1)).reshape(-1,1)

GMM = GaussianMixture(n_components=3, random_state=10)
GMM.fit(X1_sorted)

prob_X1 = GMM.predict_proba(X1_sorted)

ax1.plot(X1_sorted, prob_X1[:,0], label='Predicted Prob of x belonging to cluster 1')
ax1.plot(X1_sorted, prob_X1[:,1], label='Predicted Prob of x belonging to cluster 2')
ax1.plot(X1_sorted, prob_X1[:,2], label='Predicted Prob of x belonging to cluster 3')
ax1.scatter(2, 0.6, color='black')
ax1.scatter(2, 1.0, color='black')
ax1.plot([2, 2], [0.6, 1.0],'--', color='black')
ax1.legend()
fig1

# Ví dụ 4: Tạo dữ liệu Gaussian mixture 2D
mean = [(1,5), (2,1), (6,2)]
cov1 = np.array([[0.5, 1.0],[1.0, 0.8]])
cov2 = np.array([[0.8, 0.4],[0.4, 1.2]])
cov3 = np.array([[1.2, 1.3],[1.3, 0.9]])
cov = [cov1, cov2, cov3]
weights = [0.3, 0.3, 0.4]

X4 = plot_bivariate_mixture(means=mean, covs=cov, weights=weights, N=1000)

# Fit GMM cho dữ liệu 2D
print("The dataset we generated has a shape of", X4.shape)

gm = GaussianMixture(n_components=3, random_state=0).fit(X4)
print("Means of the 3 Gaussians fitted by GMM are\n")
print(gm.means_)

print("Covariances of the 3 Gaussians fitted by GMM are")
print(gm.covariances_)

# Vẽ clusters với covariance_type='full'
plot_gmm(GaussianMixture(n_components=3, random_state=0), X4)

# Vẽ clusters với covariance_type='tied'
plot_gmm(GaussianMixture(n_components=3, covariance_type='tied',random_state=0), X4)

# Vẽ clusters với covariance_type='diag'
plot_gmm(GaussianMixture(n_components=3, covariance_type='diag',random_state=0), X4)

# Ví dụ Image Segmentation
import urllib.request
urllib.request.urlretrieve('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/images/gauss-cat.jpeg', 'gauss-cat.jpeg')

img = plt.imread('gauss-cat.jpeg')
X = img.reshape(-1, 3)

# Segmentation với 2 clusters
n = 2
gmm = GaussianMixture(n_components=n, covariance_type='tied')
gmm.fit(X)
labels = gmm.predict(X)

seg = np.zeros(X.shape)
for label in range(n):
    seg[labels == label] = gmm.means_[label]
seg = seg.reshape(img.shape).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.title('Segmentation with 2 clusters')
plt.show()

# Segmentation với 8 clusters
n = 8
gmm = GaussianMixture(n_components=n, covariance_type='tied')
gmm.fit(X)
labels = gmm.predict(X)

seg = np.zeros(X.shape)
for label in range(n):
    seg[labels == label] = gmm.means_[label]
seg = seg.reshape(img.shape).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.title('Segmentation with 8 clusters')
plt.show()

# BÀI TẬP: Customer Segmentation
# Load dữ liệu
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/customers.csv")
print("Data shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Exercise 1: Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler

# Khởi tạo StandardScaler để chuẩn hóa dữ liệu về mean=0 và std=1
SS = StandardScaler()

# Chuẩn hóa dữ liệu: fit để học các tham số và transform để áp dụng
X = SS.fit(data).transform(data)
print("Scaled data shape:", X.shape)

# Exercise 2: Giảm chiều với PCA
from sklearn.decomposition import PCA

# Khởi tạo PCA với 2 thành phần chính
pca2 = PCA(n_components=2)

# Áp dụng PCA để giảm chiều dữ liệu
reduced_2_PCA = pca2.fit(X).transform(X)
print("PCA reduced data shape:", reduced_2_PCA.shape)
print("Explained variance ratio:", pca2.explained_variance_ratio_)

# Exercise 3: Fit GMM
# Khởi tạo mô hình Gaussian Mixture Model với 4 clusters
model = GaussianMixture(n_components=4, random_state=0)

# Huấn luyện mô hình GMM trên dữ liệu đã được giảm chiều
model.fit(reduced_2_PCA)
print("GMM fitted successfully!")
print("Means shape:", model.means_.shape)
print("Covariances shape:", model.covariances_.shape)

# Exercise 4: Dự đoán labels
# Dự đoán cluster cho từng điểm dữ liệu
PCA_2_pred = model.predict(reduced_2_PCA)
print("Predicted labels shape:", PCA_2_pred.shape)
print("Unique labels:", np.unique(PCA_2_pred))
print("Label counts:", np.bincount(PCA_2_pred))

# Exercise 4: Visualization
# Trích xuất tọa độ x và y từ dữ liệu PCA 2D
x = reduced_2_PCA[:,0]
y = reduced_2_PCA[:,1]

# Vẽ biểu đồ scatter với màu sắc theo cluster
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=PCA_2_pred, cmap='viridis', alpha=0.6)
plt.title("2D visualization of the clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster')
plt.show()

# Exercise 5: Clustering với 3 principal components
from mpl_toolkits.mplot3d import Axes3D

# PCA với 3 thành phần chính
pca3 = PCA(n_components=3)
reduced_3_PCA = pca3.fit(X).transform(X)

# Fit GMM với 3D data
mod = GaussianMixture(n_components=4, random_state=0)
PCA_3_pred = mod.fit(reduced_3_PCA).predict(reduced_3_PCA)

# Visualization 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_3_PCA[:, 0], reduced_3_PCA[:, 1], reduced_3_PCA[:, 2], c=PCA_3_pred, cmap='viridis', alpha=0.6)
ax.set_title("3D projection of the clusters")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.show()

# Phân tích kết quả clustering
print("=== PHÂN TÍCH KẾT QUẢ CLUSTERING ===")
print(f"Số lượng clusters: {len(np.unique(PCA_2_pred))}")
print(f"Số điểm dữ liệu: {len(PCA_2_pred)}")
print(f"Phân bố clusters:")
for i in range(len(np.unique(PCA_2_pred))):
    count = np.sum(PCA_2_pred == i)
    percentage = (count / len(PCA_2_pred)) * 100
    print(f"  Cluster {i}: {count} điểm ({percentage:.1f}%)")

print(f"\nExplained variance ratio (2D): {pca2.explained_variance_ratio_}")
print(f"Explained variance ratio (3D): {pca3.explained_variance_ratio_}")
