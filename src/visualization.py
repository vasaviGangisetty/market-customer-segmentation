import matplotlib.pyplot as plt


def plot_elbow(inertias):
    """
    Plot Elbow Method graph.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(inertias) + 1),
        inertias,
        marker="o"
    )

    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)

    plt.savefig("outputs/elbow_method.png")
    plt.close()


def plot_clusters(features, labels):
    """
    Plot customer clusters.
    """

    plt.figure(figsize=(8, 5))

    plt.scatter(
        features[:, 2],
        features[:, 3],
        c=labels
    )

    plt.title("Customer Clusters")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")

    plt.savefig("outputs/customer_clusters.png")
    plt.close()


def plot_pca_clusters(pca_data, labels):
    """
    Plot PCA-based clusters.
    """

    plt.figure(figsize=(8, 5))

    plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels
    )

    plt.title("PCA Customer Segmentation")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("outputs/pca_clusters.png")
    plt.close()