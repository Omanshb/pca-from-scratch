# Principal Component Analysis (PCA) from Scratch

_A complete mathematical guide with from-scratch implementation_

## 🎓 Introduction: What is PCA?

Principal Component Analysis (PCA) is one of the most fundamental dimensionality reduction techniques in machine learning and data science. At its core, PCA solves a beautiful mathematical problem: **given high-dimensional data, find the directions that capture the maximum variance**.

Think of it this way: if you're looking at a 3D object and want to capture its essence in a 2D photograph, you'd choose the angle that shows the most detail. PCA does exactly this, but mathematically and for any number of dimensions.

**The core question PCA answers**: _In what direction should I project my data so that the projected data has the largest possible variance?_

## 🧮 The Mathematical Proof: Why Eigenvectors Are Directions of Maximum Variance

### The Optimization Problem

Let's say we have mean-centered data matrix **X ∈ ℝⁿ ˣ ᵈ** and we want to find a unit vector **w ∈ ℝᵈ** that maximizes the variance of the projection **Xw**.

The variance of the projected data **Xw** is:

**Var(Xw) = (1/n) ||Xw||² = (1/n) wᵀ Xᵀ X w = wᵀ Σ w**

where **Σ = (1/n) XᵀX** is the **covariance matrix**.

So we want to solve:

**maximize: wᵀ Σ w**  
**subject to: ||w|| = 1**

### Solving with Lagrange Multipliers

Using Lagrange multipliers, we form the Lagrangian:

**L(w, λ) = wᵀ Σ w - λ(wᵀw - 1)**

Taking the gradient with respect to **w** and setting it to zero:

**∇w L = 2Σw - 2λw = 0 ⟹ Σw = λw**

**This is the definition of an eigenvector!**

### The Beautiful Result

The directions **w** that maximize variance are the **eigenvectors of the covariance matrix**, and the maximum variance achieved is the corresponding **eigenvalue λ**.

Therefore:

- **First principal component**: Eigenvector with the largest eigenvalue
- **Second principal component**: Eigenvector with the second largest eigenvalue (orthogonal to the first)
- And so on...

## 💻 Step-by-Step Implementation from Scratch

### Step 1: Data Standardization

Before applying PCA, we standardize data to prevent features with larger scales from dominating:

```python
def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data to have zero mean and unit variance.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std
```

**Why standardize?** Without this step, a feature like "salary" (range 0-100,000) would dominate "age" (range 0-100) purely due to scale differences.

### Step 2: Mean Centering

Center the data by subtracting the mean from each feature:

```python
self.mean_ = np.mean(X, axis=0)
X_centered = X - self.mean_
```

**Mathematical intuition**: Centering ensures principal components pass through the origin, making the mathematical analysis clean and interpretable.

### Step 3: Covariance Matrix Computation

Compute the covariance matrix that encodes how features vary together:

```python
cov_matrix = np.cov(X_centered.T)
```

**What this captures**: Each element **C[i,j]** represents how features **i** and **j** vary together. The diagonal elements are individual feature variances.

### Step 4: Eigenvalue Decomposition

This is where the magic happens - we find the eigenvectors and eigenvalues:

```python
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
```

**Mathematical significance**: We're solving **Σw = λw** to find the directions (**w**) of maximum variance and their magnitudes (**λ**).

### Step 5: Sorting and Selection

Sort eigenvalues in descending order to identify the most important directions:

```python
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

self.components_ = eigenvectors[:, :self.n_components].T
self.explained_variance_ = eigenvalues[:self.n_components]
```

**Key insight**: The eigenvector with the largest eigenvalue captures the direction of maximum variance. The second largest captures the direction of maximum remaining variance (orthogonal to the first).

### Step 6: Data Transformation

Project data onto the selected principal components:

```python
X_centered = X - self.mean_
X_transformed = np.dot(X_centered, self.components_.T)
```

**Geometric interpretation**: This rotates our coordinate system, expressing data in terms of the new basis vectors (principal components) rather than original features.

### Complete Implementation Structure

```python
class PCAFromScratch:
    def fit(self, X):           # Learn principal components from data
    def transform(self, X):     # Project data to PC space
    def fit_transform(self, X): # Fit and transform in one step
    def get_cumulative_variance_ratio(): # Analysis tools
```

## 📊 Understanding Explained Variance

PCA's power lies in quantifying information retention:

```python
self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
```

**Practical meaning**:

- If PC1 has explained variance ratio of 0.7, it captures 70% of total variance
- Cumulative explained variance shows total information retained with **k** components
- The "elbow method" helps choose optimal number of components

### Key Applications

**1. Dimensionality Reduction**: Reduce 1000 features to 50 components while retaining 95% of information

**2. Data Visualization**: Project high-dimensional data to 2D/3D for human interpretation

**3. Noise Filtering**: Remove components with small eigenvalues that often represent noise

**4. Feature Engineering**: Use principal components as new features that capture data's main patterns

**5. Compression**: Store data more efficiently by keeping only important components

## ⚠️ Limitations and Considerations

### 1. Linearity Assumption

PCA only captures **linear relationships**. For data with non-linear patterns, consider:

- Kernel PCA for non-linear dimensionality reduction
- t-SNE or UMAP for non-linear visualization
- Autoencoders for complex non-linear mappings

### 2. Interpretability Loss

Principal components are **combinations of all original features**, making them harder to interpret:

- PC1 might be "0.3×height + 0.4×weight + 0.2×age + ..."
- You lose the ability to say "this specific feature is important"
- Consider feature selection if interpretability is crucial

### 3. Standardization Sensitivity

The choice to standardize features dramatically impacts results:

- **With standardization**: All features have equal opportunity to influence PCs
- **Without standardization**: Large-scale features dominate
- **Decision rule**: Standardize when features have different units or scales

### 4. Outlier Sensitivity

PCA can be skewed by outliers because:

- Outliers artificially inflate variance in their direction
- This can make noise appear as a principal component
- **Solution**: Remove outliers or use robust PCA variants

### 5. Global Linear Method

PCA finds **global** linear patterns and may miss:

- **Local patterns**: Clusters or subgroups in data
- **Non-monotonic relationships**: U-shaped or cyclical patterns
- **Discrete structures**: Categories that don't form continuous clouds
