import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

#Takes in a sequences of models
class VAE_Encoder(nn.Sequential):

    #Which each model is something that reduces the data but increases its features
    def __init__(self):
        super().__init__(
            #(batch_size, Channel, Height, Weight) -> (batch_size, 128, Height, Weight)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #Input channels, output Channels  aka (batch_size, 128,  Height, Width) -> (batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            

            # (batch_size, 128, Height, Width) -> (batch_size, 128, Height /2 , Width /2
            nn.conv2d(128, 128, kernel_size=3, stride=2, padding=0), 

            #(batch_Size, 128, Height /2, Width /2) -> (batch_size, 256, Height /2, Width /2)
            VAE_ResidualBlock(128, 256),

            #(batch_size, 256, Height /2, Width /2) -> (batch_size, 256, Height /2, Width / 2)
            VAE_ResidualBlock(256, 256),

            #(batch_size, 256, Height /2, Width /2) -> (batch_size, 256, Height /4, Width /4
            nn.conv2d(256, 256, kernel_size=3, stride=2, padding=0), 

            #(batch_size, 256, Height /4, Width /4) -> (batch_size, 512, Height /4, Width /4)
            VAE_ResidualBlock(256, 512),

            #(batch_size, 512, Height /4, Width /4) -> (batch_size, 512, Height /4, Width /4)
            VAE_ResidualBlock(512, 512),

            #(batch_size, 512, Height /4, Width /4) -> (batch_size, 512, Height /8, Width /8)
            nn.conv2d(512, 512, kernel_size=3, stride=2, padding=0), 

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 512, Height /8, Width /8)
            VAE_ResidualBlock(512, 512),


            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 512, Height /8, Width /8)
            #Attention is a sequence to sequence model, does not reduce size simply connects
            VAE_AttentionBlock(512),

            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 512, Height /8, Width /8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 512, Height /8, Width /8)
            #Number of groups, number of channels/features
            nn.GrpoupNorm(32, 512),

            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 512, Height /8, Width /8)
            nn.SiLU(),

            # (batch_size, 512, Height /8, Width /8) -> (batch_size, 8, Height /8, Width /8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, Height /8, Width /8) -> (batch_size, 8, Height /8, Width /8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
     )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width)
        #Noise: (Batch_size, Channel, Height /8, Width/8), will have same size as encoder
        for module in self:

            #ONly apply to convultions with a stide of 2
            if getattr(module, "stride", None)  == (2,2):

                #Padding left, Padding right, Padding Top, Padding Bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        #(batch_size, 8, Height /8, Width /8) -> two tensors of size (batch_size, 4, Height /8, Width /8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        #Clamping the variance if its too big or small
        #(batch_size, 4, Height /8, Width /8) -> (batch_size, 4, Height /8, Width /8
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Z = N(0,1) sample -> N(mean, variance) = X
        # X = mean + stdev * z
        x = mean + stdev * noise

        #Scale the output by a constant, scaling the output by a constant
        x *= 0.18215
        return x 
