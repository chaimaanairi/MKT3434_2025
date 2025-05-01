# MKT3434 - Homework Assignment #2

## Advanced Dimensionality Reduction & Clustering GUI

### Dr. Ertuğrul BAYRAKTAR  
### Chaimaa Nairi - 23501057  

---

## Overview  
A PyQt6-based interactive application for applying and comparing dimensionality reduction techniques and clustering algorithms on datasets. Ideal for visualizing PCA, LDA, t-SNE, and UMAP projections and validating clustering with KMeans and cross-validation.

---

## Features

### Dimensionality Reduction
- **PCA** – Principal Component Analysis: Visualize explained variance by components.
- **LDA** – Linear Discriminant Analysis: Supervised 2D linear projection.
- **t-SNE** – Nonlinear embedding with adjustable perplexity.
- **UMAP** – Fast nonlinear manifold-preserving dimensionality reduction.

### Clustering
- **KMeans** clustering with dynamic cluster count selection.
- **Silhouette Score** and **Calinski-Harabasz Score** for cluster quality evaluation.

### Cross-Validation
- **K-Fold cross-validation** for clustering with adjustable data split ratios.

### Extras
- **Covariance Matrix Projection**: Simple 2D projection and eigenvector calculation demo.

---

## Comparison of Dimensionality Reduction & Evaluation Methods

| Method                | Description                                              | Pros                                                   | Cons                                                       |
|-----------------------|----------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------|
| **PCA**               | Linear projection maximizing variance                    | Fast, interpretable, unsupervised                      | Assumes linearity, ignores class labels                   |
| **LDA**               | Supervised projection maximizing class separation        | Emphasizes class distinctions                          | Requires labels, sensitive to imbalance                   |
| **t-SNE**             | Nonlinear projection preserving local structure          | Excellent for visualizing clusters                     | Sensitive to perplexity, slow for large datasets          |
| **UMAP**              | Nonlinear projection preserving global & local structure | Faster than t-SNE, good for manifold learning          | Less interpretable, parameter-sensitive                   |
| **KMeans**            | Partitioning clustering using distance minimization      | Simple, fast                                           | Requires number of clusters, sensitive to initialization  |
| **Silhouette Score**  | Measures intra- vs. inter-cluster distances              | Easy to interpret                                      | Misleading in imbalanced or overlapping data              |
| **Calinski-Harabasz** | Ratio of between- to within-cluster dispersion           | Good for compact, well-separated clusters              | Degrades with noise or elongated clusters                 |
| **K-Fold CV**         | Train/test split-based evaluation across k iterations    | Reduces overfitting risk, more robust                  | Computationally intensive, may not reflect clustering quality |

---

