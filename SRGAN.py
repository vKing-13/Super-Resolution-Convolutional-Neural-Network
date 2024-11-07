import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# Dataset class to prepare low-resolution and high-resolution image pairs
class HRDataset(Dataset):
    def __init__(self, hr_image_path, upscaling_factor=4, patch_size=96, transform=None):
        self.hr_image_path = hr_image_path
        self.upscaling_factor = upscaling_factor
        self.patch_size = patch_size
        self.transform = transform
        self.filenames = [os.path.join(hr_image_path, x) for x in os.listdir(hr_image_path) if x.endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_image = Image.open(self.filenames[idx]).convert("YCbCr")
        y, _, _ = hr_image.split()

        # Crop to patch size
        hr_patch = y.crop((0, 0, self.patch_size, self.patch_size))
        lr_patch = hr_patch.resize(
            (self.patch_size // self.upscaling_factor, self.patch_size // self.upscaling_factor),
            Image.BICUBIC
        ).resize((self.patch_size, self.patch_size), Image.BICUBIC)

        if self.transform:
            hr_patch = self.transform(hr_patch)
            lr_patch = self.transform(lr_patch)

        return lr_patch, hr_patch

# residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# generator network
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        '''
            we use two convolutional layers with small 3X3 kernels and 64 feature
            maps followed by batch-normalization layers [32] and ParametricReLU
        '''
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        # residual block
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.final = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residual_blocks(initial) + initial
        x = self.upsample(x)
        return self.final(x)

# discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        '''
            The discriminator network is trained to solve
            the maximization problem in Equation 2. It contains eight
            convolutional layers with an increasing number of 3x3
            filter kernels, increasing by a factor of 2 from 64 to 512 kernels
            as in the VGG network [49]. Strided convolutions are
            used to reduce the image resolution each time the number
            of features is doubled. The resulting 512 feature maps are
            followed by two dense layers and a final sigmoid activation
            function to obtain a probability for sample classification.
        '''
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # k3n64s1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n64s2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n128s1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n128s2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n256s1
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n256s2
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n512s1
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # k3n512s2
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            # Dense(1024)
            nn.Linear(512 * 6 * 6, 1024),
            # LeakyReLU
            nn.LeakyReLU(0.2, inplace=True),
            # Dense(1)
            nn.Linear(1024, 1),
            # sigmoid activation function
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# content los
class VGGContentLoss(nn.Module):
    def __init__(self):
        super(VGGContentLoss, self).__init__()
        vgg = vgg19(weights="IMAGENET1K_V1").features[:9].eval()  # Updated for torchvision 0.13+
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, gen_images, hr_images):
        # Repeat the single channel to create a 3-channel image
        gen_images_3c = gen_images.repeat(1, 3, 1, 1)
        hr_images_3c = hr_images.repeat(1, 3, 1, 1)

        if gen_images_3c.size() != hr_images_3c.size():
            gen_images_3c = F.interpolate(gen_images_3c, size=hr_images_3c.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute the content loss between the VGG features of generated and HR images
        return nn.functional.mse_loss(self.vgg(gen_images_3c), self.vgg(hr_images_3c))

# training
def train_srgan(generator, discriminator, dataloader, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    vgg_content_loss = VGGContentLoss().to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for lr_images, hr_images in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            disc_optimizer.zero_grad()
            real_loss = nn.functional.binary_cross_entropy_with_logits(
                discriminator(hr_images), torch.ones(hr_images.size(0), 1).to(device)
            )
            fake_images = generator(lr_images).detach()
            fake_loss = nn.functional.binary_cross_entropy_with_logits(
                discriminator(fake_images), torch.zeros(fake_images.size(0), 1).to(device)
            )
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()

            gen_optimizer.zero_grad()
            gen_images = generator(lr_images)

            # adversarial loss
            adv_loss = nn.functional.binary_cross_entropy_with_logits(
                discriminator(gen_images), torch.ones(gen_images.size(0), 1).to(device)
            )
            content_loss = vgg_content_loss(gen_images, hr_images)
            gen_loss = content_loss + 1e-3 * adv_loss
            gen_loss.backward()
            gen_optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")


generator = Generator()
discriminator = Discriminator()

hr_image_path = "./HighResolutionImg"
transform = ToTensor()
train_dataset = HRDataset(hr_image_path, upscaling_factor=4, patch_size=96, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

train_srgan(generator, discriminator, dataloader, num_epochs=100)


test_image_path = "./Set14/baboon.png"
result_dir = "./Result"
os.makedirs(result_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)  

# validation
def test_srgan(generator, image_path, result_dir, upscaling_factor=4):
    image = Image.open(image_path).convert("YCbCr")
    y, cb, cr = image.split()

    lr_y = y.resize((y.width // upscaling_factor, y.height // upscaling_factor), Image.BICUBIC)
    lr_y_upscaled = lr_y.resize((y.width, y.height), Image.BICUBIC)  

    downsampled_image = Image.merge("YCbCr", [lr_y_upscaled, cb.resize(lr_y_upscaled.size, Image.BICUBIC), cr.resize(lr_y_upscaled.size, Image.BICUBIC)]).convert("RGB")
    downsampled_image.save(os.path.join(result_dir, "Downsampled_baboon.png"))

    transform = ToTensor()
    lr_y_tensor = transform(lr_y_upscaled).unsqueeze(0).to(device) 

    with torch.no_grad():
        sr_y_tensor = generator(lr_y_tensor).squeeze(0)  

    sr_y_image = sr_y_tensor.cpu().clamp(0, 1).squeeze().numpy() * 255
    sr_y_image = Image.fromarray(sr_y_image.astype('uint8'), mode='L')

    cb_upscaled = cb.resize(sr_y_image.size, Image.BICUBIC)
    cr_upscaled = cr.resize(sr_y_image.size, Image.BICUBIC)

    sr_image = Image.merge("YCbCr", [sr_y_image, cb_upscaled, cr_upscaled]).convert("RGB")

    sr_image.save(os.path.join(result_dir, "Reconstructed_baboon.png"))

    print("Downsampled and super-resolved images saved to:", result_dir)

test_srgan(generator, test_image_path, result_dir, upscaling_factor=4)