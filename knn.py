import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.spatial import distance
from collections import Counter
from multiprocessing import Pool

def knn_search(train, train_labels, query, k):
    """Output a predication label for a query input.
    
    Args:
        train (np.array): trainig samples.
        train_labels (np.array): labels for the training samples.
        query (np.array): new sample.
        k (int): number of neighbors to use.
        
    Returns:
        object: The predicted label.
    """
    distances = np.array([distance.euclidean(train_sample, query) for train_sample in train])
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = train_labels[k_nearest_indices]
    chosen_label_item, = Counter(k_nearest_labels).most_common(1)
    chosen_label, _ = chosen_label_item
    
    return chosen_label

def plot_acc_as_k(train, train_labels, test, test_labels):
    print("acc as k", flush=True)
    k_ranges = range(1, 1001)
    items = [(train, train_labels, test, test_labels, k) for k in k_ranges]

    print("Starting...", flush=True)
    with Pool() as pool:
        results = pool.starmap(knn_accuracy, items)

    plt.figure()
    plt.scatter(k_ranges, results)
    plt.xlabel("k value")
    plt.ylabel("prediction accuracy")
    plt.grid(True)
    plt.show()

def plot_acc_as_n(test, test_labels):
    print("acc as n", flush=True)
    mnist = fetch_openml("mnist_784", as_frame=False)
    data = mnist["data"]
    labels = mnist["target"]
    idx = np.random.RandomState(0).choice(70000, 11000)
    n_ranges = range(100, 5001, 100)
    items = ((data[idx[:n], :].astype(int), labels[idx[:n]], test, test_labels, 1) for n in n_ranges)

    print("Starting...", flush=True)
    with Pool() as pool:
        results = pool.starmap(knn_accuracy, items)

    plt.figure()
    plt.scatter(n_ranges, results)
    plt.xlabel("n value")
    plt.ylabel("prediction accuracy")
    plt.grid(True)
    plt.show()

def knn_accuracy(train, train_labels, test, test_labels, k):
    """Output the accuracy of k-nn.
    
    Args:
        train (np.array): trainig samples.
        train_labels (np.array): labels for the training samples.
        test (np.array): test samples.
        test_labels (np.array): labels for the test samples
        k (int): number of neighbors to use.
        
    Returns:
        int: K-nn accuracy.
    """
    prediction = np.array([knn_search(train, train_labels, query=query, k=k) for query in test])
    acc = sum(prediction == test_labels) / len(test_labels)
    
    return acc

if __name__ == "__main__":
    mnist = fetch_openml("mnist_784", as_frame=False)
    data = mnist["data"]
    labels = mnist["target"]

    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:1000], :].astype(int)
    train_labels = labels[idx[:1000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    print("Starting...", flush=True)
    answer = input("0 for plot_acc_as_k else for plot_acc_as_n: ")
    if answer == "0":
        plot_acc_as_k(train, train_labels, test, test_labels)

    else:
        plot_acc_as_n(test, test_labels)