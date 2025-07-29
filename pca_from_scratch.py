"""
Principal Component Analysis (PCA) Implementation from Scratch

This module contains a complete implementation of PCA without using
scikit-learn's PCA class, built from fundamental linear algebra operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class PCAFromScratch:
    """
    Principal Component Analysis implementation from scratch.
    
    This class implements PCA using fundamental linear algebra operations:
    1. Data standardization
    2. Covariance matrix computation
    3. Eigenvalue decomposition
    4. Principal component selection
    5. Data transformation
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components to keep.
                         If None, keep all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
    def fit(self, X: np.ndarray) -> 'PCAFromScratch':
        """
        Fit PCA on the data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            self: Returns the instance itself
        """
        X = np.array(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape
        
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        cov_matrix = np.cov(X_centered.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimensional space.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=np.float64)
        
        X_centered = X - self.mean_
        
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform the data in one step.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            Cumulative explained variance ratio
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def plot_explained_variance(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot explained variance ratio and cumulative explained variance.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        components_range = range(1, len(self.explained_variance_ratio_) + 1)
        ax1.bar(components_range, self.explained_variance_ratio_)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance Ratio')
        ax1.grid(True, alpha=0.3)
        
        cumulative_var = self.get_cumulative_variance_ratio()
        ax2.plot(components_range, cumulative_var, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data to have zero mean and unit variance.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        
    Returns:
        X_standardized: Standardized data
        mean: Mean of original data
        std: Standard deviation of original data
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    std = np.where(std == 0, 1, std)
    
    X_standardized = (X - mean) / std
    
    return X_standardized, mean, std 