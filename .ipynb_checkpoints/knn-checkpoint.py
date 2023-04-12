from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.datasets import fetch_openml


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


def knn_accuracy(train, train_labels, test, test_labels, k):
    """Output the accuracy of k-nn.
    
    Args:
        train (np.array): trainig samples.
        train_labels (np.array): labels for the training samples.
        test (np.array): test samples.
        test_labels (np.array): labels for the test samples
        k (int): number of neighbors to use.
        
    Returns:
        intt: K-nn accuracy.
    """
    prediction = np.array([knn_search(train, train_labels, query=query, k=k) for query in test])
    acc = sum(prediction == test_labels) / len(test_labels)
    
    return acc
