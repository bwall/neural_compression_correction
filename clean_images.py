#!/usr/bin/python3
import argparse
import torchvision.transforms as transforms
import torch.utils.data
from neural_compression_correction.models import load_dn4

from tqdm import tqdm

from PIL import Image
import os


def mirror_directory_structure(original_base, new_base):
    original_base = original_base.rstrip(os.sep) + os.sep
    new_base = new_base.rstrip(os.sep) + os.sep
    
    if not os.path.exists(new_base):
        os.makedirs(new_base)
    for root, dirs, _ in os.walk(original_base):
        for dr in dirs:
            p = os.path.join(root.replace(original_base, new_base), dr)
            os.makedirs(p)
            

def list_file_paths(base_path):
    paths = []
    for root, dirs, fns in os.walk(base_path):
        for fn in fns:
            paths.append(os.path.join(root, fn))
    return paths


class RawPathedImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)
        self.length = len(self.image_paths)
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path, new_path = self.image_paths[idx]
        oimage = self.to_tensor(Image.open(path).convert("RGB"))
        
        return (new_path, oimage)

            
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Neural Compression Correction')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for inference (default: 1)')
    parser.add_argument('-w', '--workers', type=int, default=1, metavar='N',
                        help='Number of input workers (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to file or directory of images to process")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to file or directory to output results to")

    args = parser.parse_args()
    
    batch_size = args.batch_size
    single_file = os.path.isfile(args.input)
    if not single_file:
        paths = list_file_paths(args.input)
        output_path = args.output.rstrip(os.sep) + os.sep
        mirror_directory_structure(args.input, args.output)
        paths = [
            (path, os.path.join(path.replace(args.input, output_path))) for path in paths
        ]
    else:
        paths = [args.input, args.output]
        batch_size = 1
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    loader = torch.utils.data.DataLoader(
        RawPathedImageDataset(paths),
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    model = load_dn4(device=device)
    unloader = transforms.ToPILImage()
    
    with torch.no_grad():
        
        for new_paths, image_batch in tqdm(loader):
            preds = model(image_batch.to(device)).cpu()
            for i, np in enumerate(new_paths):
                img = unloader(preds[i, :, :, :])
                img.save(np)
        
if __name__ == '__main__':
    main()