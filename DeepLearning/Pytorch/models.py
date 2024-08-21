from torch import nn

class FashionModelV0(nn.Module):

    def __init__(self, img_width:int, img_height:int, hidden_units:int, output_shape:int):
        super().__init__()
        
        self.input_shape = img_width * img_height

        self.layer_stack = nn.Sequential(
                nn.Flatten(), # Flattening layer: Used to convert the input tensor from (batch_size, 1, H, W) to (batch_size, 1, H*W)
                nn.Linear(in_features=self.input_shape, out_features=hidden_units), # Linear Projection layer (aka Fully conntected layer)
                nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

    def forward(self, x):
        return self.layer_stack(x)