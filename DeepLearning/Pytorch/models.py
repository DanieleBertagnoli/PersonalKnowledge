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