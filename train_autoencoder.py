import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from unet.unet_model import UNet

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
unet = UNet(n_channels=1)
unet.load_state_dict(torch.load('./models/unet_1500.pth'))
unet = unet.train()
unet = unet.cuda()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

epochs = 10
step = 1500

for epoch in range(epochs):
    for data in train_dataloader:
        data = [d.cuda() for d in data]
        images, _ = data
        optimizer.zero_grad()
        output = unet(images)
        loss = torch.nn.functional.mse_loss(output, images)
        loss.backward()
        optimizer.step()

        # generate and save the output image
        if step % 100 == 0:
            import matplotlib.pyplot as plt
            import numpy as np
            output = output[0].cpu().detach().numpy()
            output = np.squeeze(output)
            plt.imshow(output, cmap='gray')
            plt.savefig(f'./images/unet_{step}.png')

        step += 1
        print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        if step % 500 == 0:
            torch.save(unet.state_dict(), f'./models/unet_{step}.pth')




