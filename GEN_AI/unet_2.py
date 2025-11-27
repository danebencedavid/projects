import os
import json
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_DIR = "/workspace/data/deepglobe/train"
CACHE_DIR_SUFFIX = "_cache"
CHECKPOINTS_DIR = "/workspace/checkpoints"
OUTPUTS_DIR = "/workspace/outputs"
LOGS_DIR = "/workspace/training_logs"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

N_CLASSES = 7

IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 120               
LR = 2e-4
BETA1 = 0.5
LAMBDA_L1 = 100.0
SAVE_INTERVAL = 5

RESUME = True                  
RESUME_EPOCH = 95             

torch.manual_seed(42)

class DeepGlobeDataset(Dataset):
    def __init__(self, folder, crop_size=IMG_SIZE, crops_per_image=10, use_jitter=True):
        self.folder = folder
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.use_jitter = use_jitter

        self.cache_folder = folder + CACHE_DIR_SUFFIX
        os.makedirs(self.cache_folder, exist_ok=True)

        self.to_tensor = T.ToTensor()

        self.color_to_class = {
            (0, 0, 0): 0,
            (0, 255, 255): 1,
            (255, 255, 0): 2,
            (255, 0, 255): 3,
            (0, 255, 0): 4,
            (0, 0, 255): 5,
            (255, 255, 255): 6,
        }

        sat_files = sorted([f for f in os.listdir(folder) if f.endswith("_sat.jpg")])
        self.images = []
        self.mask_cache_paths = []

        print("Caching DeepGlobe masks (first run only)...")
        for sat in tqdm(sat_files, desc="Masks"):
            base = sat.replace("_sat.jpg", "")
            sat_path = os.path.join(folder, sat)
            mask_path = os.path.join(folder, base + "_mask.png")
            cache_path = os.path.join(self.cache_folder, base + "_mask.npy")

            if not os.path.exists(cache_path):
                mask_img = Image.open(mask_path).convert("RGB")
                mask_np = np.array(mask_img)
                H, W, _ = mask_np.shape
                class_mask = np.zeros((H, W), dtype=np.uint8)
                for rgb, c in self.color_to_class.items():
                    class_mask[np.all(mask_np == rgb, axis=-1)] = c
                np.save(cache_path, class_mask)

            self.images.append(sat_path)
            self.mask_cache_paths.append(cache_path)

        self.sample_idx = []
        for i in range(len(self.images)):
            for _ in range(self.crops_per_image):
                self.sample_idx.append(i)

        print(f"Dataset ready: {len(self.images)} images {len(self.sample_idx)} samples")

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        img_idx = self.sample_idx[idx]
        img = Image.open(self.images[img_idx]).convert("RGB")
        mask = np.load(self.mask_cache_paths[img_idx])  

        resize_size = int(self.crop_size * 1.12) if self.use_jitter else self.crop_size
        img = TF.resize(img, (resize_size, resize_size), antialias=True)

        mask_img = Image.fromarray(mask)
        mask = np.array(mask_img.resize((resize_size, resize_size), resample=Image.NEAREST))

        if self.use_jitter and resize_size > self.crop_size:
            i, j, h, w = T.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            mask = mask[i:i+h, j:j+w]
        else:
            img = TF.resize(img, (self.crop_size, self.crop_size))
            if mask.shape != (self.crop_size, self.crop_size):
                mask_img = Image.fromarray(mask)
                mask = np.array(mask_img.resize((self.crop_size, self.crop_size),
                                                resample=Image.NEAREST))

        if self.use_jitter and torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask = np.fliplr(mask).copy()

        img_t = self.to_tensor(img) * 2 - 1

        mask_tensor = torch.from_numpy(mask).long()
        mask_onehot = torch.zeros((N_CLASSES, self.crop_size, self.crop_size),
                                  dtype=torch.float32)
        mask_onehot.scatter_(0, mask_tensor.unsqueeze(0), 1.0)

        return img_t, mask_onehot


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, drop=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if drop: layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], 1)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, ngf=64):
        super().__init__()
        self.d1 = UNetDown(in_channels, ngf, norm=False)
        self.d2 = UNetDown(ngf, ngf*2)
        self.d3 = UNetDown(ngf*2, ngf*4)
        self.d4 = UNetDown(ngf*4, ngf*8)
        self.d5 = UNetDown(ngf*8, ngf*8)
        self.d6 = UNetDown(ngf*8, ngf*8)
        self.d7 = UNetDown(ngf*8, ngf*8)
        self.d8 = UNetDown(ngf*8, ngf*8, norm=False)

        self.u1 = UNetUp(ngf*8,     ngf*8)
        self.u2 = UNetUp(ngf*16,    ngf*8)
        self.u3 = UNetUp(ngf*16,    ngf*8)
        self.u4 = UNetUp(ngf*16,    ngf*8)
        self.u5 = UNetUp(ngf*16,    ngf*4)
        self.u6 = UNetUp(ngf*8,     ngf*2)
        self.u7 = UNetUp(ngf*4,     ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2)
        d4 = self.d4(d3); d5 = self.d5(d4); d6 = self.d6(d5)
        d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8, d7); u2 = self.u2(u1, d6); u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4); u5 = self.u5(u4, d3); u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=10, ndf=32):
        super().__init__()
        def block(i, o, s=2, norm=True):
            layers=[nn.Conv2d(i,o,4,s,1,bias=False)]
            if norm: layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        seq=[]
        seq += block(in_channels, ndf, norm=False)
        seq += block(ndf, ndf*2)
        seq += block(ndf*2, ndf*4)
        seq += block(ndf*4, ndf*8, s=1)
        seq += [nn.Conv2d(ndf*8, 1, 4, 1, 1, bias=False)]
        self.model = nn.Sequential(*seq)

    def forward(self, img, mask):
        return self.model(torch.cat([img, mask], 1))


def save_checkpoint(epoch, G, D, optG, optD, hist):
    torch.save(G.state_dict(), os.path.join(CHECKPOINTS_DIR, f"G_{epoch}.pt"))
    torch.save(D.state_dict(), os.path.join(CHECKPOINTS_DIR, f"D_{epoch}.pt"))
    torch.save({
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "hist": hist
    }, os.path.join(CHECKPOINTS_DIR, f"state_{epoch}.pt"))
    print(f"Checkpoint saved: epoch {epoch}")

    with open(os.path.join(LOGS_DIR, "loss_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

def train():
    dataset = DeepGlobeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    optG = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    optD = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    start_epoch = 0
    hist = {"G": [], "D": []}

    if RESUME:
        print(f"Resuming from epoch {RESUME_EPOCH}...")
    
        G.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, f"G_{RESUME_EPOCH}.pt")))
        D.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, f"D_{RESUME_EPOCH}.pt")))
    
        state_path = os.path.join(CHECKPOINTS_DIR, f"state_{RESUME_EPOCH}.pt")
        state = torch.load(state_path)
    
        if "optG" in state:
            optG.load_state_dict(state["optG"])
        if "optD" in state:
            optD.load_state_dict(state["optD"])
    
        if "hist" in state:
            hist = state["hist"]
        else:
            print("no hist.")
            hist = {"G": [], "D": []}
    
        start_epoch = RESUME_EPOCH
    
    else:
        G.apply(init_weights)
        D.apply(init_weights)

    fixed_real, fixed_mask = next(iter(loader))
    fixed_real = fixed_real[:4].to(DEVICE)
    fixed_mask = fixed_mask[:4].to(DEVICE)

    print("Training started...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        g_loss_sum = 0
        d_loss_sum = 0

        for real, mask in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):

            real = real.to(DEVICE)
            mask = mask.to(DEVICE)

            # -------------- D ----------------
            D.zero_grad()

            real_logits = D(real, mask)
            real_labels = torch.full_like(real_logits, 0.9)
            d_real = BCE(real_logits, real_labels)

            with torch.no_grad():
                fake = G(mask)

            fake_logits = D(fake, mask)
            fake_labels = torch.zeros_like(fake_logits)
            d_fake = BCE(fake_logits, fake_labels)

            d_loss = (d_real + d_fake) * 0.5
            d_loss.backward()
            optD.step()

            # -------------- G ----------------
            G.zero_grad()
            fake = G(mask)
            pred_fake = D(fake, mask)

            g_adv = BCE(pred_fake, real_labels)
            g_l1 = L1(fake, real) * LAMBDA_L1

            if epoch < 2:
                g_loss = g_l1
            else:
                g_loss = g_adv + g_l1

            g_loss.backward()
            optG.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        avgG = g_loss_sum / len(loader)
        avgD = d_loss_sum / len(loader)
        hist["G"].append(avgG)
        hist["D"].append(avgD)

        print(f"Epoch {epoch+1}: D={avgD:.4f}, G={avgG:.4f}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(epoch + 1, G, D, optG, optD, hist)

            G.eval()
            with torch.no_grad():
                fake = G(fixed_mask).detach()
                real_vis = (fixed_real.detach().cpu() + 1) / 2
                fake_vis = (fake.detach().cpu() + 1) / 2
                
                grid = make_grid(torch.cat([real_vis, fake_vis], dim=0), nrow=4)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_image(grid, f"{OUTPUTS_DIR}/epoch_{epoch+1}_{ts}.png")
            G.train()

    save_checkpoint(NUM_EPOCHS, G, D, optG, optD, hist)
    print("Training done.")


if __name__ == "__main__":
    train()
