# MODULE 2: DISTANCE METRICS - CODE ONLY

# Import các thư viện cần thiết
import warnings
warnings.warn = lambda *args, **kwargs: None

import pandas as pd
import numpy as np
import scipy
from scipy.spatial.distance import euclidean, cityblock, cosine
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
%matplotlib inline

# Hàm tính average distance giữa hai tập dữ liệu
def avg_distance(X1, X2, distance_func):
    from sklearn.metrics import jaccard_score
    res = 0
    for x1 in X1:
        for x2 in X2:
            if distance_func == jaccard_score: # jaccard_score chỉ trả về similarity
                res += 1 - distance_func(x1, x2)
            else:
                res += distance_func(x1, x2)
    return res / (len(X1) * len(X2))

# Hàm tính pairwise distance
def avg_pairwise_distance(X1, X2, distance_func):
    return sum(map(distance_func, X1, X2)) / min(len(X1), len(X2))

# Load dữ liệu Iris
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/iris.csv')
print("Iris dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Loại bỏ cột petal_width
df.drop(['petal_width'], axis=1, inplace=True)
print("\nAfter dropping petal_width:")
print(df.head())

# Xem các loài hoa
species = df['species'].unique()
print(f"\nSpecies: {species}")

# Tạo dữ liệu cho từng loài
attrs = ['sepal_length', 'sepal_width', 'petal_length']
setosa_data = df.loc[df['species'] == 'setosa'][attrs].to_numpy()
versicolor_data = df.loc[df['species'] == 'versicolor'][attrs].to_numpy()
virginica_data = df.loc[df['species'] == 'virginica'][attrs].to_numpy()

print(f"\nSetosa data shape: {setosa_data.shape}")
print(f"Versicolor data shape: {versicolor_data.shape}")
print(f"Virginica data shape: {virginica_data.shape}")

# Visualization 3D
from mpl_toolkits.mplot3d import Axes3D
markers = ['o', 'v', '^']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for specie, marker in zip(species, markers):
    specie_data = df.loc[df['species'] == specie][attrs]
    xs, ys, zs = [specie_data[attr] for attr in attrs]
    ax.scatter(xs, ys, zs, marker=marker, label=specie)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.legend()
plt.title('3D Visualization of Iris Species')
plt.show()

# EUCLIDEAN DISTANCE
print("\n=== EUCLIDEAN DISTANCE ===")
# Ví dụ đơn giản
print(f"Euclidean distance between [0,0] and [3,4]: {euclidean([0, 0], [3, 4])}")

# Tính average euclidean distance giữa các loài
setosa_versicolor_euclidean = avg_distance(setosa_data, versicolor_data, euclidean)
setosa_virginica_euclidean = avg_distance(setosa_data, virginica_data, euclidean)

print(f"Average euclidean distance between setosa and versicolor: {setosa_versicolor_euclidean:.4f}")
print(f"Average euclidean distance between setosa and virginica: {setosa_virginica_euclidean:.4f}")

# Sử dụng sklearn pairwise distances
from sklearn.metrics.pairwise import paired_euclidean_distances

X = np.array([[0, 0]], dtype=float)
Y = np.array([[3, 4]], dtype=float)
print(f"Paired euclidean distance: {paired_euclidean_distances(X, Y).mean():.4f}")
print(f"Our function result: {avg_pairwise_distance(X, Y, euclidean):.4f}")

# Pairwise distances cho dữ liệu thực
row_dist = paired_euclidean_distances(setosa_data, versicolor_data)
print(f"Row-wise euclidean distances mean: {row_dist.mean():.4f}")

# MANHATTAN DISTANCE
print("\n=== MANHATTAN DISTANCE ===")
# Ví dụ đơn giản
print(f"Manhattan distance between [1,1] and [-2,2]: {cityblock([1, 1], [-2, 2])}")

# Tính average manhattan distance giữa các loài
setosa_setosa_manhattan = avg_distance(setosa_data, setosa_data, cityblock)
setosa_versicolor_manhattan = avg_distance(setosa_data, versicolor_data, cityblock)
setosa_virginica_manhattan = avg_distance(setosa_data, virginica_data, cityblock)

print(f"Average manhattan distance between setosa and setosa: {setosa_setosa_manhattan:.4f}")
print(f"Average manhattan distance between setosa and versicolor: {setosa_versicolor_manhattan:.4f}")
print(f"Average manhattan distance between setosa and virginica: {setosa_virginica_manhattan:.4f}")

# Sử dụng sklearn pairwise distances
from sklearn.metrics.pairwise import manhattan_distances
X = np.array([[1, 1]])
Y = np.array([[-2, 2]])
print(f"Manhattan distances matrix:\n{manhattan_distances(X, Y)}")

# COSINE DISTANCE
print("\n=== COSINE DISTANCE ===")
# Ví dụ đơn giản
print(f"Cosine distance between [1,1] and [-1,-1]: {cosine([1, 1], [-1, -1])}")

# Load dữ liệu auto-mpg
df_auto = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/auto-mpg.data',
    header=None, delim_whitespace=True,
    names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Xử lý dữ liệu
df_auto['car_name'] = df_auto['car_name'].str.split(n=1).apply(lambda lst: lst[0]).replace('chevrolet', 'chevy')
df_auto.rename(columns={'car_name': 'make'}, inplace=True)
df_auto = df_auto[['mpg', 'weight', 'make']]

# Normalize dữ liệu
dfn = df_auto[['mpg', 'weight']]
df_auto[['mpg', 'weight']] = (dfn-dfn.min())/(dfn.max()-dfn.min())

print("\nAuto dataset after preprocessing:")
print(df_auto.head())

# So sánh chevy và honda
chevy = df_auto.loc[df_auto['make'] == 'chevy']
honda = df_auto.loc[df_auto['make'] == 'honda']

plt.figure(figsize=(10, 6))
plt.scatter(chevy['mpg'], chevy['weight'], marker='o', label='chevy', alpha=0.7)
plt.scatter(honda['mpg'], honda['weight'], marker='^', label='honda', alpha=0.7)
plt.xlabel('mpg (normalized)')
plt.ylabel('weight (normalized)')
plt.title('Chevy vs Honda Cars')
plt.legend()
plt.show()

# Tính cosine distance
chevy_data = chevy[['mpg', 'weight']].to_numpy()
honda_data = honda[['mpg', 'weight']].to_numpy()

chevy_chevy_cosine = avg_distance(chevy_data, chevy_data, cosine)
honda_honda_cosine = avg_distance(honda_data, honda_data, cosine)
honda_chevy_cosine = avg_distance(honda_data, chevy_data, cosine)

print(f"Average cosine distance between chevy and chevy: {chevy_chevy_cosine:.4f}")
print(f"Average cosine distance between honda and honda: {honda_honda_cosine:.4f}")
print(f"Average cosine distance between honda and chevy: {honda_chevy_cosine:.4f}")

# Sử dụng sklearn pairwise distances
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

X = np.array([[1, 1]])
Y = np.array([[-1, -1]])
print(f"Cosine distances: {cosine_distances(X, Y)}")
print(f"Cosine similarity: {cosine_similarity(X, Y)}")
print(f"Cosine distance from similarity: {1-cosine_similarity(X,Y)}")

# DBSCAN với các distance metrics khác nhau
print("\n=== DBSCAN WITH DIFFERENT DISTANCE METRICS ===")
from sklearn.cluster import DBSCAN

# Load synthetic clustering data
df_synthetic = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/synthetic_clustering.csv')

plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(df_synthetic['x'], df_synthetic['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data')

# DBSCAN with Euclidean
plt.subplot(1, 3, 2)
dbscan_euclidean = DBSCAN(eps=0.1, metric=euclidean)
dbscan_euclidean.fit(df_synthetic)
colors = np.random.random(size=3*(dbscan_euclidean.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df_synthetic['x'], df_synthetic['y'], c=[colors[l] for l in dbscan_euclidean.labels_])
plt.title('DBSCAN with Euclidean Distance')

# DBSCAN with Manhattan
plt.subplot(1, 3, 3)
dbscan_manhattan = DBSCAN(eps=0.1, metric=cityblock)
dbscan_manhattan.fit(df_synthetic)
colors = np.random.random(size=3*(dbscan_manhattan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df_synthetic['x'], df_synthetic['y'], c=[colors[l] for l in dbscan_manhattan.labels_])
plt.title('DBSCAN with Manhattan Distance')

plt.tight_layout()
plt.show()

# JACCARD DISTANCE
print("\n=== JACCARD DISTANCE ===")
from sklearn.metrics import jaccard_score

# Load breast cancer data
df_cancer = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/breast-cancer.data',
    header=None,
    names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])

print("Breast cancer dataset shape:", df_cancer.shape)
print("\nAge groups:")
print(sorted(df_cancer['age'].unique()))
print(df_cancer.age.value_counts())

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
OH = OneHotEncoder()
X_cancer = OH.fit_transform(df_cancer.loc[:, df_cancer.columns != 'age']).toarray()
print(f"\nOne-hot encoded data shape: {X_cancer.shape}")

# Phân tích Jaccard distance giữa các nhóm tuổi
X30to39 = X_cancer[df_cancer[df_cancer.age == '30-39'].index]
X60to69 = X_cancer[df_cancer[df_cancer.age == '60-69'].index]

print(f"\nAge group 30-39 shape: {X30to39.shape}")
print(f"Age group 60-69 shape: {X60to69.shape}")

# Tính Jaccard distances
jaccard_30_30 = avg_distance(X30to39, X30to39, jaccard_score)
jaccard_60_60 = avg_distance(X60to69, X60to69, jaccard_score)
jaccard_30_60 = avg_distance(X30to39, X60to69, jaccard_score)

print(f"Average Jaccard distance between 30-39 and 30-39: {jaccard_30_30:.4f}")
print(f"Average Jaccard distance between 60-69 and 60-69: {jaccard_60_60:.4f}")
print(f"Average Jaccard distance between 30-39 and 60-69: {jaccard_30_60:.4f}")

# BÀI TẬP
print("\n=== BÀI TẬP ===")

# Exercise 1: Jaccard distance
sentence1 = 'Hello everyone and welcome to distance metrics'
sentence2 = 'Hello world and welcome to distance metrics'

s1set = set(sentence1.split())
s2set = set(sentence2.split())
jaccard_similarity = len(s1set.intersection(s2set)) / len(s1set.union(s2set))
jaccard_distance = 1 - jaccard_similarity

print(f"Exercise 1 - Jaccard Distance:")
print(f"Các từ trong câu 1: {s1set}")
print(f"Các từ trong câu 2: {s2set}")
print(f"Từ chung: {s1set.intersection(s2set)}")
print(f"Tất cả các từ: {s1set.union(s2set)}")
print(f"Jaccard Distance: {jaccard_distance:.4f}")

# Exercise 2: Euclidean và Manhattan distance
p1 = np.array([4, -3, 1])
p2 = np.array([-5, 1, -7])

euclidean_dist = scipy.spatial.distance.euclidean(p1, p2)
manhattan_dist = scipy.spatial.distance.cityblock(p1, p2)
difference = abs(manhattan_dist - euclidean_dist)

print(f"\nExercise 2 - Distance Comparison:")
print(f"Điểm p1: {p1}")
print(f"Điểm p2: {p2}")
print(f"Euclidean distance: {euclidean_dist:.4f}")
print(f"Manhattan distance: {manhattan_dist:.4f}")
print(f"Absolute difference: {difference:.4f}")

# Exercise 3: Cosine distance
p1_cosine = np.array([1, 2, 3]).reshape(1, -1)
p2_cosine = np.array([-2, -4, -6]).reshape(1, -1)

cosine_dist = cosine_distances(p1_cosine, p2_cosine)

print(f"\nExercise 3 - Cosine Distance:")
print(f"Vector p1: {p1_cosine}")
print(f"Vector p2: {p2_cosine}")
print(f"Cosine distance: {cosine_dist[0][0]:.4f}")
print("Giải thích: p1 và p2 ngược hướng nhau (p2 = -2*p1), nên cosine distance = 2")

# Exercise 4: Pairwise distances
X1_pairwise = np.arange(8).reshape(4, 2)
X2_pairwise = np.arange(8)[::-1].reshape(4, 2)

paired_euclidean = sklearn.metrics.pairwise.paired_euclidean_distances(X1_pairwise, X2_pairwise)
paired_manhattan = sklearn.metrics.pairwise.paired_manhattan_distances(X1_pairwise, X2_pairwise)

print(f"\nExercise 4 - Pairwise Distances:")
print(f"X1:\n{X1_pairwise}")
print(f"X2:\n{X2_pairwise}")
print(f"Paired Euclidean distances: {paired_euclidean}")
print(f"Paired Manhattan distances: {paired_manhattan}")

# Tổng kết
print("\n=== TỔNG KẾT ===")
print("Đã hoàn thành tất cả các bài tập về Distance Metrics:")
print("1. Jaccard Distance - So sánh độ tương tự giữa các tập hợp")
print("2. Euclidean Distance - Khoảng cách vật lý")
print("3. Manhattan Distance - Khoảng cách tổng tuyệt đối")
print("4. Cosine Distance - Khoảng cách góc giữa các vector")
print("5. Pairwise Distances - Khoảng cách từng cặp điểm")
print("6. DBSCAN với các distance metrics khác nhau")
