from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt
import umap


class AdvancedDRValidationTab(QWidget):
    """
    A QWidget tab that allows users to run and visualize various
    dimensionality reduction and clustering algorithms.
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout(self)

        # Create all sections
        self.create_pca_section()
        self.create_lda_section()
        self.create_tsne_section()
        self.create_umap_section()
        self.create_kmeans_section()
        self.create_cv_section()
        self.create_projection_demo()

    # ---------------- PCA Section ---------------- #

    def create_pca_section(self):
        """Creates UI for PCA configuration and execution."""
        group = QGroupBox("PCA - Principal Component Analysis")
        layout = QHBoxLayout()
        self.pca_components = QSpinBox()
        self.pca_components.setRange(1, 50)

        btn = QPushButton("Run PCA")
        btn.clicked.connect(self.run_pca)

        layout.addWidget(QLabel("n_components:"))
        layout.addWidget(self.pca_components)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_pca(self):
        """Performs PCA and plots explained variance."""
        n = self.pca_components.value()
        pca = PCA(n_components=n)
        X = self.parent.X_train
        pca.fit(X)

        self.parent.figure.clear()
        ax = self.parent.figure.add_subplot(111)
        ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax.set_title("Explained Variance by PCA")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        self.parent.canvas.draw()

    # ---------------- LDA Section ---------------- #

    def create_lda_section(self):
        """Creates UI for LDA execution."""
        group = QGroupBox("LDA - Linear Discriminant Analysis")
        layout = QHBoxLayout()

        btn = QPushButton("Run LDA")
        btn.clicked.connect(self.run_lda)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_lda(self):
        """Performs LDA and visualizes 2D projection."""
        lda = LDA(n_components=2)
        X_r2 = lda.fit_transform(self.parent.X_train, self.parent.y_train)

        self.parent.figure.clear()
        ax = self.parent.figure.add_subplot(111)

        for label in np.unique(self.parent.y_train):
            idx = self.parent.y_train == label
            ax.scatter(X_r2[idx, 0], X_r2[idx, 1], label=str(label))

        ax.set_title("LDA Projection")
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.legend()
        self.parent.canvas.draw()

    # ---------------- t-SNE Section ---------------- #

    def create_tsne_section(self):
        """Creates UI for t-SNE parameter input and execution."""
        group = QGroupBox("t-SNE")
        layout = QHBoxLayout()

        self.tsne_perplexity = QDoubleSpinBox()
        self.tsne_perplexity.setRange(5, 100)
        self.tsne_perplexity.setValue(30)

        btn = QPushButton("Run t-SNE")
        btn.clicked.connect(self.run_tsne)

        layout.addWidget(QLabel("Perplexity:"))
        layout.addWidget(self.tsne_perplexity)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_tsne(self):
        """Performs t-SNE and visualizes the result."""
        tsne = TSNE(n_components=2,
                    perplexity=self.tsne_perplexity.value(),
                    random_state=42)
        X_embedded = tsne.fit_transform(self.parent.X_train)

        self.parent.figure.clear()
        ax = self.parent.figure.add_subplot(111)

        for label in np.unique(self.parent.y_train):
            idx = self.parent.y_train == label
            ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=str(label))

        ax.set_title("t-SNE Projection")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend()
        self.parent.canvas.draw()

    # ---------------- UMAP Section ---------------- #

    def create_umap_section(self):
        """Creates UI for UMAP execution."""
        group = QGroupBox("UMAP (Fast DR Alternative to t-SNE)")
        layout = QHBoxLayout()

        btn = QPushButton("Run UMAP")
        btn.clicked.connect(self.run_umap)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_umap(self):
        """Performs UMAP and visualizes the result."""
        reducer = umap.UMAP(random_state=42)
        X_reduced = reducer.fit_transform(self.parent.X_train)

        self.parent.figure.clear()
        ax = self.parent.figure.add_subplot(111)

        for label in np.unique(self.parent.y_train):
            idx = self.parent.y_train == label
            ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=str(label))

        ax.set_title("UMAP Projection")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend()
        self.parent.canvas.draw()

    # ---------------- K-Means Section ---------------- #

    def create_kmeans_section(self):
        """Creates UI for KMeans clustering execution."""
        group = QGroupBox("K-Means Clustering")
        layout = QHBoxLayout()

        self.k_clusters = QSpinBox()
        self.k_clusters.setRange(1, 20)

        btn = QPushButton("Run Clustering")
        btn.clicked.connect(self.run_kmeans)

        layout.addWidget(QLabel("# Clusters:"))
        layout.addWidget(self.k_clusters)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_kmeans(self):
        """Performs KMeans clustering and evaluates metrics."""
        k = self.k_clusters.value()
        model = KMeans(n_clusters=k)
        y_pred = model.fit_predict(self.parent.X_train)

        sil_score = silhouette_score(self.parent.X_train, y_pred)
        ch_score = calinski_harabasz_score(self.parent.X_train, y_pred)

        self.parent.metrics_text.setText(
            f"Silhouette: {sil_score:.4f}, CH Score: {ch_score:.2f}"
        )

    # ---------------- Cross-Validation Section ---------------- #

    def create_cv_section(self):
        """Creates UI for cross-validation configuration and execution."""
        group = QGroupBox("Cross-Validation")
        layout = QHBoxLayout()

        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 20)

        self.split_selector = QComboBox()
        self.split_selector.addItems(["70-15-15", "80-10-10", "60-20-20"])
        self.split_selector.setToolTip("Data split ratio for Train/Val/Test")

        btn = QPushButton("Run CV")
        btn.clicked.connect(self.run_cv)

        layout.addWidget(QLabel("k-folds:"))
        layout.addWidget(self.cv_folds)
        layout.addWidget(QLabel("Split:"))
        layout.addWidget(self.split_selector)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def run_cv(self):
        """Runs KFold cross-validation on KMeans clustering."""
        k = self.cv_folds.value()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kf.split(self.parent.X_train):
            X_t = self.parent.X_train[train_idx]
            X_v = self.parent.X_train[test_idx]
            y_t = self.parent.y_train[train_idx]
            y_v = self.parent.y_train[test_idx]

            clf = KMeans(n_clusters=len(np.unique(y_t)))
            clf.fit(X_t)
            pred = clf.predict(X_v)

            # Simplified accuracy proxy — assumes labels ≈ clusters
            acc = np.mean(pred == y_v)
            scores.append(acc)

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        self.parent.metrics_text.setText(
            f"CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}"
        )

    # ---------------- Eigenvector Demo ---------------- #

    def create_projection_demo(self):
        """Creates UI for simple 1D covariance matrix projection."""
        group = QGroupBox("Covariance Matrix Projection (1D)")
        layout = QHBoxLayout()

        btn = QPushButton("Compute Eigenvector")
        btn.clicked.connect(self.compute_eigenvector)
        layout.addWidget(btn)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def compute_eigenvector(self):
        """Computes and displays the top eigenvector of a 2x2 matrix."""
        Sigma = np.array([[5, 2], [2, 3]])
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        top_vector = eigvecs[:, idx[0]]

        self.parent.metrics_text.setText(
            f"Top-1 Eigenvector (1D): {top_vector}"
        )
