import torch
from torch import nn

#   _      _                         __  __           _      _ 
#  | |    (_)                       |  \/  |         | |    | |
#  | |     _ _ __   ___  __ _ _ __  | \  / | ___   __| | ___| |
#  | |    | | '_ \ / _ \/ _` | '__| | |\/| |/ _ \ / _` |/ _ \ |
#  | |____| | | | |  __/ (_| | |    | |  | | (_) | (_| |  __/ |
#  |______|_|_| |_|\___|\__,_|_|    |_|  |_|\___/ \__,_|\___|_|


class FashionModelLinear(nn.Module):

    def __init__(self, img_width:int, img_height:int, hidden_units:int, output_shape:int):
        super().__init__()
        
        self.input_shape = img_width * img_height

        self.layer_stack = nn.Sequential(
                nn.Flatten(), # Flattening layer: Used to convert the input tensor from (batch_size, 1, H, W) to (batch_size, 1*H*W)
                nn.Linear(in_features=self.input_shape, out_features=hidden_units), # Linear Projection layer (aka Fully conntected layer)
                nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

    def forward(self, x):
        return self.layer_stack(x)
    


#   _   _                   _      _                         __  __           _      _ 
#  | \ | |                 | |    (_)                       |  \/  |         | |    | |
#  |  \| | ___  _ __ ______| |     _ _ __   ___  __ _ _ __  | \  / | ___   __| | ___| |
#  | . ` |/ _ \| '_ \______| |    | | '_ \ / _ \/ _` | '__| | |\/| |/ _ \ / _` |/ _ \ |
#  | |\  | (_) | | | |     | |____| | | | |  __/ (_| | |    | |  | | (_) | (_| |  __/ |
#  |_| \_|\___/|_| |_|     |______|_|_| |_|\___|\__,_|_|    |_|  |_|\___/ \__,_|\___|_|


class FashionModelNonLinear(nn.Module):

    def __init__(self, img_width:int, img_height:int, hidden_units:int, output_shape:int):
        super().__init__()
        
        self.input_shape = img_width * img_height

        self.layer_stack = nn.Sequential(
                nn.Flatten(), # Flattening layer: Used to convert the input tensor from (batch_size, 1, H, W) to (batch_size, 1*H*W)
                nn.Linear(in_features=self.input_shape, out_features=hidden_units), # Linear Projection layer (aka Fully conntected layer)
                nn.ReLU(), # Non-linear activation function
                nn.Linear(in_features=hidden_units, out_features=output_shape),
                nn.ReLU()
            )

    def forward(self, x):
        return self.layer_stack(x)
    


#    _____ _   _ _   _             _                        _    __  __           _      _     
#   / ____| \ | | \ | |           | |                      | |  |  \/  |         | |    | |    
#  | |    |  \| |  \| |  ______   | |__   __ _ ___  ___  __| |  | \  / | ___   __| | ___| |___ 
#  | |    | . ` | . ` | |______|  | '_ \ / _` / __|/ _ \/ _` |  | |\/| |/ _ \ / _` |/ _ \ / __|
#  | |____| |\  | |\  |           | |_) | (_| \__ \  __/ (_| |  | |  | | (_) | (_| |  __/ \__ \
#   \_____|_| \_|_| \_|           |_.__/ \__,_|___/\___|\__,_|  |_|  |_|\___/ \__,_|\___|_|___/
#


class FashionModelCnnV0(nn.Module):

    def __init__(self, img_channels:int, hidden_units:int, output_shape:int):
        super().__init__()

        # The model must accept only Nx28x28 images

        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1), 
            nn.ReLU(), # Non-linear activation function
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) # Takes 2x2 kernel and keeps only the highest value (trivially the data dimensions are divided by 2)
        ) # 256x14x14

        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        ) #256x7x7

        self.classification_layer = nn.Sequential(
            nn.Flatten(), # 256*49
            nn.Linear(in_features=hidden_units*49, out_features=output_shape)
        ) 

    def forward(self, x):
        if x.shape[2] != 28 or x.shape[3] != 28:
            print(f'ERROR! The model can accept only 28x28 images.')
            exit(1)

        y = self.convolutional_block_1(x)
        y = self.convolutional_block_2(y)
        y = self.classification_layer(y)
        return y
    

class IntelClassificationV0(nn.Module):

    def __init__(self, img_channels:int, output_shape:int, hidden_units:int):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) # Image size 64x64
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*2,
                      out_channels=hidden_units*2,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) # Image size 32x32
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2,
                      out_channels=hidden_units*4,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*4,
                      out_channels=hidden_units*4,
                      kernel_size=(3,3),
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) # Image size 16x16
        )

        self.classification_layer = nn.Sequential(
            nn.Flatten(), # hidden_dim*16*16
            nn.Linear(in_features=hidden_units * 4 * 16 * 16, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(in_features=512, out_features=output_shape)
        )


    def forward(self, x):

        if x.shape[2] != 128 or x.shape[3] != 128:
            print(f'ERROR! The model can accept only 128x128 images.')
            exit(1)

        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classification_layer(x)
        return x
    



#  _    _      _                     
# | |  | |    | |                    
# | |__| | ___| |_ __   ___ _ __ ___ 
# |  __  |/ _ \ | '_ \ / _ \ '__/ __|
# | |  | |  __/ | |_) |  __/ |  \__ \
# |_|  |_|\___|_| .__/ \___|_|  |___/
#               | |                  
#               |_|                  


from tqdm import tqdm

from evaluation import get_accuracy

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
            "model_test_loss": test_loss.to('cpu').item(),
            "model_test_accuracy": test_acc
        }
    


def train_model(model: torch.nn.Module, 
                loss_fn: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                device: str,
                epochs: int,
                optimizer: torch.optim.Optimizer):
    """
    Trains a PyTorch model over a specified number of epochs and evaluates it at the end of each epoch.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        loss_fn (torch.nn.Module): The loss function used to compute the loss during training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing data.
        device (str): The device to use for training (e.g., 'cpu' or 'cuda').
        epochs (int): The number of epochs to train the model.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
    
    Returns:
        None: The function prints out the training loss and evaluation metrics at the end of each epoch.
    """

    # Calculate total number of batches (epochs * batches per epoch)
    total_batches = epochs * len(train_dataloader)

    epochs_test_results = []

    for epoch in range(epochs):

        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)):
            model.train()
            
            # Move data to the correct device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            predictions = model(images)
            
            # Loss calculation
            loss = loss_fn(predictions, labels)
            train_loss += loss.item()  # accumulate the loss for this batch

            # Optimizer zero grad
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Optimizer optimization 
            optimizer.step()
        
        # Adjust the training loss by averaging over the number of batches
        train_loss = train_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}')
        
        # Testing and evaluation
        test_results = evaluate_model(model, loss_fn, test_dataloader, device)

        print(f'Epoch {epoch+1}/{epochs} - Test Loss: {test_results["model_test_loss"]:.4f}, Accuracy: {test_results["model_test_accuracy"]:.4f}%')
