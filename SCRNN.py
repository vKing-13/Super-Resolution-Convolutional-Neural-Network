import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# dataset
class HRDataset(Dataset):
    '''
    The upscaling factor is 3.
    Whereas the ImageNet provides over 5 million sub-images even using a stride of 33.
    Thus the 91-image dataset can be decomposed into 24,800 sub-images, which are extracted from original images with a stride of 14
    '''
    def __init__(self, hr_image_path, upscaling_factor=3, patch_size=33, stride=14, transform=None):
        self.hr_image_path = hr_image_path
        self.upscaling_factor = upscaling_factor
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.filenames = [os.path.join(hr_image_path, x) for x in os.listdir(hr_image_path) if x.endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.filenames) * self.get_patch_count(self.filenames[0])

    def __getitem__(self, i):
        image_index = i // self.get_patch_count(self.filenames[0])
        patch_index = i % self.get_patch_count(self.filenames[0])

        hr_image = Image.open(self.filenames[image_index])
        # convert unproper imagess to YCbCr format
        if hr_image.mode != "YCbCr":
            hr_image = hr_image.convert("YCbCr")

        # only get the luminance channel
        y, _, _ = hr_image.split()

        # Extract patch from luminance channel
        hr_patch = self.extract_patch(y, patch_index)

        # downsampling
        lr_patch = hr_patch.resize(
            (hr_patch.width // self.upscaling_factor, hr_patch.height // self.upscaling_factor),
            Image.BICUBIC
        ).resize((hr_patch.width, hr_patch.height), Image.BICUBIC)

        # transforms
        if self.transform:
            hr_patch = self.transform(hr_patch)
            lr_patch = self.transform(lr_patch)
        return lr_patch, hr_patch

    def get_patch_count(self, filename):
        img = Image.open(filename)
        # convert unproper imagess to YCbCr format
        if img.mode != "YCbCr":
            img = img.convert("YCbCr")

        y, _, _ = img.split()

        patches_x = (y.width - self.patch_size) // self.stride + 1
        patches_y = (y.height - self.patch_size) // self.stride + 1
        return patches_x * patches_y

    def extract_patch(self, img, patch_index):
        patches_x = (img.width - self.patch_size) // self.stride + 1
        x_index = patch_index % patches_x
        y_index = patch_index // patches_x
        x_start = x_index * self.stride
        y_start = y_index * self.stride
        return img.crop((x_start, y_start, x_start + self.patch_size, y_start + self.patch_size))




hr_image_path = "./HighResolutionImg"
# hr_image_path = "./high_res" #56000
# hr_image_path = "./HRImg"
# hr_image_path = "./DIV2K_train/dataset1"
validation_path = "./Set5"
save_path = "./Saved"

os.makedirs(save_path, exist_ok=True)

# transform
transform = transforms.Compose([transforms.ToTensor()])

# dataloader
dataset = HRDataset(hr_image_path, upscaling_factor=3, patch_size=33, stride=14, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # padding (0,0,0) err, proceed on ((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# init device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(torch.cuda.is_available())
# device="cuda"
print(f"Using {device} device")
model = SRCNN().to(device)

# loss and optimizer
criterion = nn.MSELoss()
# learning rate 0.0001
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    # Use tqdm to create a progress bar for batches
    for lr_images, hr_images in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # resolve conv2d error
        if lr_images.dim() == 3:
            lr_images = lr_images.unsqueeze(1)
        if hr_images.dim() == 3:
            hr_images = hr_images.unsqueeze(1)

        # forward pass
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # show progress
        tqdm.write(f'Batch Loss: {loss.item():.4f}')

    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# calc psnr value
def calculate_psnr(output, target):
    mse = nn.functional.mse_loss(output.unsqueeze(0), target)
    psnr = 10 * torch.log10(1 / mse)
    return psnr

# testing n result
def validate_and_save(model, validation_path, save_path, upscaling_factor=3):
    model.eval()
    transform = transforms.ToTensor()

    for img_name in os.listdir(validation_path):
        img_path = os.path.join(validation_path, img_name)
        img = Image.open(img_path).convert("YCbCr")
        y, cb, cr = img.split()

        # low res img
        lr_y = y.resize((y.width // upscaling_factor, y.height // upscaling_factor), Image.BICUBIC)
        lr_y = lr_y.resize((y.width, y.height), Image.BICUBIC)

         # Save the downsampled image
        downsampled_image = Image.merge("YCbCr", [lr_y, cb.resize(lr_y.size, Image.BICUBIC), cr.resize(lr_y.size, Image.BICUBIC)]).convert("RGB")
        downsampled_image.save(os.path.join(save_path, f"Bicubic_{img_name}"))
        
        lr_y_tensor = transform(lr_y).unsqueeze(0).to(device)
        hr_y_tensor = transform(y).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).squeeze(0).unsqueeze(0)

        # get PSNR
        psnr = calculate_psnr(sr_y_tensor, hr_y_tensor).item()
        print(f"{img_name} - PSNR: {psnr:.2f} dB")

        # convert to img
        sr_y_image = sr_y_tensor.cpu().squeeze().clamp(0, 1).numpy() * 255
        sr_y_image = Image.fromarray(sr_y_image.astype('uint8'), mode='L')

        # upscale the cb cr, merge
        cb = cb.resize(sr_y_image.size, Image.BICUBIC)
        cr = cr.resize(sr_y_image.size, Image.BICUBIC)
        sr_image = Image.merge("YCbCr", [sr_y_image, cb, cr]).convert("RGB")

        # save
        sr_image.save(os.path.join(save_path, f"SCRNN_{img_name}"))

    model.train()

validate_and_save(model, validation_path, save_path, upscaling_factor=3)