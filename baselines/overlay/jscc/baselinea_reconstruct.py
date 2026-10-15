# baselinea_reconstruct.py — reconstruct a split with a trained JSCC-CNN checkpoint.
# Run from the jscc dir so model.py / channel.py resolve to this repo.
# Saves uint8 PNGs (same quantization as metrics.py as_img_array).

import os
import argparse
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import OrderedDict

from model import DeepJSCC
from utils import set_seed

IMG = 224


class ImageOnly(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        self.t = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.t(Image.open(self.paths[idx]).convert("RGB")), os.path.basename(self.paths[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--c', type=int, required=True)
    ap.add_argument('--snr', type=float, required=True)
    ap.add_argument('--channel', default='Rayleigh', choices=['AWGN', 'Rayleigh'])
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DeepJSCC(c=args.c, channel_type=args.channel, snr=args.snr, P_encoder=1.0).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    if next(iter(sd)).startswith('module.'):
        sd = OrderedDict((k[7:], v) for k, v in sd.items())
    model.load_state_dict(sd)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = ImageOnly(args.input_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    to_pil = ToPILImage()
    n = 0
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            out = model(images).clamp(0, 1).cpu()
            for i in range(out.shape[0]):
                name = os.path.splitext(names[i])[0] + '.png'
                to_pil(out[i]).save(os.path.join(args.output_dir, name))
                n += 1
    print(f"reconstructed {n} images -> {args.output_dir}")


if __name__ == '__main__':
    main()
