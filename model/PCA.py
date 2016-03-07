import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density=0.4, random_state=42)
X=X.toarray()
n=1
pca = PCA(n_components=n)
pca.fit(X)

energy=pca.explained_variance_ratio_.sum() 
print energy
while (energy<0.8):
    n=n+1
    pca = PCA(n_components=n)
    pca.fit(X)
    energy=pca.explained_variance_ratio_.sum() 
    print energy
new=pca.fit_transform(X, y=None)
print new