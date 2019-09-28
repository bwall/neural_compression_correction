import torch
import torch.nn as nn
import torch.nn.functional as F


class DecompressNet0(nn.Module):
    """Only works with images with even dimensions"""
    def __init__(self):
        super(DecompressNet0, self).__init__()
        upscale_factor = 2
        self.decompress = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            
            nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),          
            
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(x) + x, min_val=0.0, max_val=1.0)

    
class DecompressNet2(nn.Module):
    """Only works with images with even dimensions"""
    def __init__(self):
        super(DecompressNet2, self).__init__()
        upscale_factor = 2
        self.decompress = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(8, 8), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            
            nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),          
            
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(x) + x, min_val=0.0, max_val=1.0)
    

class ParallelResidual(nn.Module):
    def __init__(self, in_features):
        super(ParallelResidual, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.ReLU(True),
            
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.Dropout(0.1),
        )
            
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
            nn.Dropout(0.1),
            nn.ReLU(True),
            
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
            nn.Dropout(0.1),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
            nn.Dropout(0.1),
            nn.ReLU(True),
            
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        return torch.tanh(x + self.conv3(x) + self.conv7(x) + self.conv5(x))

    
class DecompressNet4(nn.Module):
    def __init__(self):
        super(DecompressNet4, self).__init__()

        self.decompress = nn.Sequential(
            nn.Dropout(0.1),
            
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
            nn.ReLU(True),
            
            ParallelResidual(128),
            nn.Dropout(0.1),
            ParallelResidual(128),
            nn.Dropout(0.1),
            ParallelResidual(128),
            nn.Dropout(0.1),
            ParallelResidual(128),
            nn.Dropout(0.1),           
            
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),          
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(x) + x, min_val=0.0, max_val=1.0)
    
    
def load_dn4(device="cuda", path="models/jpeg_to_raw.dn4.1569400880.model"):
    dn = DecompressNet4().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()

