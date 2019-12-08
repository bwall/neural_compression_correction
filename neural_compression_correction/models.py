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
    

class Residual(nn.Module):
    def __init__(self, in_features, kernel_size=3):
        super(Residual, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        return torch.tanh(x + self.conv(x))


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
    

class DecompressNet9(nn.Module):
    def __init__(self):
        super(DecompressNet9, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            Residual(128, 3),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 7),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 3),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),    
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            Residual(128, 3),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 7),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 3),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),    
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(self.attention(x) * x) + x, min_val=0.0, max_val=1.0)
    
    
class DecompressNet10(nn.Module):
    def __init__(self):
        super(DecompressNet10, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            Residual(128, 3),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 7),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 3),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),   
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            Residual(128, 3),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 7),
            nn.Dropout(0.1),
            
            Residual(128, 5),
            nn.Dropout(0.1),
            
            Residual(128, 3),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(min_val=0.0, max_val=1.0),    
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(self.attention(x) * x) + x, min_val=0.0, max_val=1.0)
    
    
class DParallelResidual(nn.Module):
    def __init__(self, in_features):
        super(DParallelResidual, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
        )
            
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
        )
        
    def forward(self, x):
        return torch.tanh(x + self.conv3(x) + self.conv7(x) + self.conv5(x))
    
# ResidualLayer(3).cuda()(torch.rand(16, 3, 128, 128).cuda()).size()
    
    
class DecompressNetNA13(nn.Module):
    def __init__(self):
        super(DecompressNetNA13, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            
            DParallelResidual(128),
            DParallelResidual(128),
            DParallelResidual(128),
            DParallelResidual(128),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),          
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(x) + x, min_val=0.0, max_val=1.0)


class DecompressNetNA14(nn.Module):
    def __init__(self):
        super(DecompressNetNA14, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),

            DParallelResidual(128),
            DParallelResidual(128),
            DParallelResidual(128),
            DParallelResidual(128),
            DParallelResidual(128),

            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),
        )

    def forward(self, x):
        return F.hardtanh(self.decompress(x) + x, min_val=0.0, max_val=1.0)


class TParallelResidual(nn.Module):
    def __init__(self, in_features):
        super(TParallelResidual, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
        )
            
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_features, in_features, kernel_size=5, padding=2),
        )
        
    def forward(self, x):
        return torch.relu(x + self.conv3(x) + self.conv7(x) + self.conv5(x))
    


class SuperResolutionNetNA14(nn.Module):
    def __init__(self):
        super(SuperResolutionNetNA14, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            
            TParallelResidual(128),
            TParallelResidual(128),
            TParallelResidual(128),
            
            nn.Conv2d(128, 12, kernel_size=1, stride=1, padding=0),

            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.decompress(x)

class SuperResolutionNetNA15(nn.Module):
    def __init__(self):
        super(SuperResolutionNetNA15, self).__init__()

        self.decompress = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            
            TParallelResidual(128),
            TParallelResidual(128),
            TParallelResidual(128),
            TParallelResidual(128),
            
            nn.Conv2d(128, 12, kernel_size=1, stride=1, padding=0),

            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.decompress(x)

    
def load_dn4(device="cuda", path="models/jpeg_to_raw.dn4.1569400880.model"):
    dn = DecompressNet4().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()


def load_dn14na(device="cuda", path="models/jpeg_to_raw.dn14na_jpeg_quiet_focused.latest.model"):
    dn = DecompressNetNA14().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()


def load_dn10_sharpen(device="cuda", path="models/jpeg_to_raw.dn10_sharpen_quiet_focused.1571452521.model"):
    dn = DecompressNet10().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()


def load_dn9(device="cuda", path="models/jpeg_to_raw.dn9_quiet_focused.1570854406.model"):
    dn = DecompressNet9().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()

def load_dn13na(device="cuda", path="models/jpeg_to_raw.dn13na_jpeg_quiet_super_stdfocused.1573791000.model"):
    dn = DecompressNetNA13().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()
    

def load_sr14na(device="cuda", path="models/jpeg_to_raw.sr14na_jpeg.latest.model"):
    dn = SuperResolutionNetNA14().to(device=device)
    dn.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    return dn.to(device=device).eval()
    
