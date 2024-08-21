import torch
import torch.utils
import torch.utils.data 

def get_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates the accuracy of predictions.

    Parameters:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels (as class indices).

    Returns:
        float: The accuracy of the predictions as a percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def evaluate_model(model: torch.nn.Module,
                   loss_fn: torch.nn.Module,
                   test_dataloader: torch.utils.data.DataLoader,
                   device: str):
    """
    Evaluates the model's performance on a test dataset.

    Parameters:
        model (torch.nn.Module): The neural network model to be evaluated.
        loss_fn (torch.nn.Module): The loss function used to compute the model's loss.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
        device (str): The device to perform the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the model name, average loss, and accuracy on the test set.
    """
    
    test_loss, test_acc = 0.0, 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.inference_mode():  # Disable gradient computation for inference

        for images, labels in test_dataloader:
            # Move data to the correct device
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute predictions
            predictions = model(images) 
            
            # Loss and accuracy calculation
            test_loss += loss_fn(predictions, labels)
            test_acc += get_accuracy(labels, predictions.argmax(dim=1))

        # Calculate the average loss and accuracy
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

        return {
            "model_name": model.__class__.__name__,
            "model_loss": test_loss.to('cpu').item(),
            "model_accuracy": test_acc
        }
