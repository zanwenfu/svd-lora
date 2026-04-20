import torch
import torch.nn as nn

class extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
          nn.Conv2d(1,16,5,1,2), 
          nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(         
          nn.Conv2d(16, 32, 5, 1, 2),     
          nn.ReLU(), nn.MaxPool2d(2))
        
        self.dim_redu = nn.Sequential(nn.Linear(32*7*7,256),nn.Tanh())
    def forward(self, x):
        N,C,W,H = x.shape
        assert C == 1 and W == 28 and H == 28, 'input tensor should be of dimension N*1*28*28'
        
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       

        return self.dim_redu(x)

  