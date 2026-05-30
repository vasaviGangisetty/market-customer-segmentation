from sklearn.decomposition import PCA


def apply_pca(features):
    """
    Reduce dimensions for visualization.
    """

    pca = PCA(n_components=2, random_state=42)

    pca_data = pca.fit_transform(features)

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    return pca_data