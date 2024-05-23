import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from unet.unet_model import UNet

val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]))
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
unet = UNet(n_channels=1)
unet = unet.cuda()

unet.load_state_dict(torch.load('./models/unet_2500.pth'))
unet = unet.eval()

data = next(iter(val_dataloader))
data = [d.cuda() for d in data]
images, _ = data

output = unet(images)
import matplotlib.pyplot as plt
import numpy as np
output = output[0].cpu().detach().numpy()
output = np.squeeze(output)
plt.imshow(output, cmap='gray')
plt.savefig(f'./images/unet_val_pred.png')
gt_image = images[0].cpu().detach().numpy()
gt_image = np.squeeze(gt_image)
plt.imshow(gt_image, cmap='gray')
plt.savefig(f'./images/unet_val_gt.png')
