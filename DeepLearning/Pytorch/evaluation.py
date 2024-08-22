import torch
import torch.utils
import torch.utils.data 

def get_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates the accuracy of predictions.

    Parameters:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.

    Returns:
        float: The accuracy of the predictions as a percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



def get_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, classes: int, confusion_matrix: list = None) -> list:
    """
    Calculates the confusion matrix of the predictions.

    Parameters:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.
        classes (int): The number of classes.
        confusion_matrix (list, optional): An existing confusion matrix to update. If None, a new matrix will be created.

    Returns:
        list: The confusion matrix as a list of lists.
    """

    # Initialize confusion matrix if not provided
    if confusion_matrix is None:
        confusion_matrix = [[0 for j in range(classes)] for i in range(classes)]

    # Populate the confusion matrix
    for index, predicted_label in enumerate(y_pred):
        gt_label = y_true[index]
        confusion_matrix[gt_label.item()][predicted_label.item()] += 1

    return confusion_matrix


import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix: list, class_names: list, title: str = "Confusion Matrix"):
    """
    Plots the confusion matrix.

    Parameters:
        confusion_matrix (list): The confusion matrix as a list of lists.
        class_names (list): The list of class names corresponding to the classes in the confusion matrix.
        title (str): The title of the plot.
    """
    # Convert the confusion matrix to a NumPy array
    cm_array = np.array(confusion_matrix)

    # Create the figure and axis with larger size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the confusion matrix using imshow
    cax = ax.matshow(cm_array, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set the title and labels
    ax.set_title(title, pad=20)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Set the ticks and labels with proper rotation
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, ha='center')  # Rotate labels to 90 degrees
    ax.set_yticklabels(class_names)

    # Annotate the cells with the number of instances
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm_array[i, j], ha='center', va='center', color='black')

    # Adjust layout to make room for labels
    plt.tight_layout()

    # Show the plot
    plt.show()