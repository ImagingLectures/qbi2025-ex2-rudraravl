from typing import Callable
import numpy as np
import albumentations as A
from sklearn.neighbors import KNeighborsClassifier
import sklearn

def create_dataset_subset(data: np.ndarray, labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Implement the function to create a subset of the dataset with n samples maintaining the distribution of labels.

    Args:
        data (np.ndarray): 
        labels (np.ndarray): 
        n (int): 

    Returns:
        tuple[np.ndarray, np.ndarray]: 
    """
    new_data, new_labels = sklearn.utils.resample(data, labels, n_samples=n, stratify=labels)
    return (new_data, new_labels)
    

def augment_data(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Implement the function to augment the dataset by applying the following transformations:
    - RandomCrop
    - HorizontalFlip
    - VerticalFlip
    - RandomBrightnessContrast
    - ShiftScaleRotate

    Args:
        data (np.ndarray): 
        labels (np.ndarray): 

    Returns:
        tuple[np.ndarray, np.ndarray]: 
    """
    crop = A.Compose([
        A.RandomCrop(width=64, height=64)
    ])
    h_flip = A.Compose([
        A.HorizontalFlip(p=1)
    ])
    v_flip = A.Compose([
        A.VerticalFlip(p=1)
    ])
    brightness_contrast = A.Compose([
        A.RandomBrightnessContrast(p=1)
    ])
    shift_scale_rotate = A.Compose([
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-30, 30), p=1)
    ])

    cropped = np.array([crop(image=image)['image'] for image in data])
    h_flipped = np.array([h_flip(image=image)['image'] for image in data])
    v_flipped = np.array([v_flip(image=image)['image'] for image in data])
    brightness_contrasted = np.array([brightness_contrast(image=image)['image'] for image in data])
    shifted_scaled_rotated = np.array([shift_scale_rotate(image=image)['image'] for image in data])

    augmented_data = np.concatenate((cropped, h_flipped, v_flipped, brightness_contrasted, shifted_scaled_rotated), axis=0)
    augmented_labels = np.concatenate([labels for _ in range(5)], axis=0)

    return (augmented_data, augmented_labels)


def split_train_test_dataset(data: np.ndarray, labels: np.ndarray, percentage: float = 0.8, shuffle: bool = False) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implement the function to split the dataset into training and testing sets.

    Args:
        data (np.ndarray): 
        labels (np.ndarray): 
        percentage (float): 
        shuffle (bool):

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: 
    """
    train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=1-percentage, shuffle=shuffle)
    return (train_data, train_labels), (test_data, test_labels)


def train_kNN(data: np.ndarray, labels: np.ndarray, k: int) -> KNeighborsClassifier:
    """
    Implement the function to train kNN classifiers.

    Args:
        data (np.ndarray): 
        labels (np.ndarray): 
        k (int): 

    Returns: kNN Classifier
        KNeighborsClassifier
    """
    data = data.reshape(data.shape[0], -1)
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(data, labels)
    return kNN
    

def predict_kNN(kNN: KNeighborsClassifier, data: np.ndarray) -> np.ndarray:
    """
    Implement the function to predict the labels of the data using the trained kNN classifier.

    Args:
        kNN (KNeighborsClassifier): 
        data (np.ndarray): 

    Returns:
        np.ndarray: 
    """
    data = data.reshape(data.shape[0], -1)
    return np.array(kNN.predict(data))
    

def evaluate_predictions(ground_truth: np.ndarray, labels: np.ndarray, metric: Callable) -> float:
    """
    Implement the function to evaluate the predictions using the given metric.

    Args:
        ground_truth (np.ndarray): 
        labels (np.ndarray): 
        metric (Callable): 

    Returns:
        float: 
    """
    return metric(ground_truth, labels)
    