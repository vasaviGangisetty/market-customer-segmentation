from sklearn.cluster import KMeans


def find_elbow(features):
    """
    Calculate inertia values for Elbow Method.
    Automatically handles small datasets.
    """

    inertias = []

    max_clusters = min(10, len(features))

    for k in range(1, max_clusters + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(features)
        inertias.append(model.inertia_)

    return inertias


def apply_kmeans(features, n_clusters=3):
    """
    Apply KMeans clustering.
    """

    # Prevent cluster count from exceeding data size
    n_clusters = min(n_clusters, len(features))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features)

    return labels, model