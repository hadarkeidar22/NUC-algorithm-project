from setups import MyDataset
from transformations import myTransforms
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from my_model import Autoencoder
from torchvision.transforms import ToTensor, ToPILImage
SAMPLE = 'sample'
LABEL = 'label'
FPN = 'FPN'

if not os.path.exists('./denoised_images'):
    os.mkdir('./denoised_images')


num_epochs = 2
batch_size = 1
learning_rate = 0.0008
img_dir = r'path'

def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))

def to_img(x):
    x = x.view(x.size(0), 1, 512, 640)
    return x

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

def tensor_round(tensor):
    return torch.round(tensor)



dataset = MyDataset(img_dir= img_dir, transforms=myTransforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
dataset_size = len(dataset)
loss_values = []
last_10_mean_loss_values = []
loss_per_epoch = []
plt.plot(np.array(loss_values), 'r')
min_loss = 1
this_noise = None
for epoch in range(num_epochs):
    index = 1
    for data in dataloader:
        if this_noise == None:
            orig_image, noise = data[LABEL], data[FPN]
            this_noise = noise
        else:
            orig_image = data[LABEL]
        noised_image = np.add(orig_image, this_noise)
        noised_image = np.clip(noised_image, 0, 1)
        print('epoch number: '+str(epoch+1)+' out of '+str(num_epochs)+', in process: '+str(format((batch_size*100*(index/dataset_size)), '.2f'))+"%")
        index = index + 1
        # ===================forward=====================
        output = model(orig_image)
        loss = criterion(output, orig_image)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        if (loss.data.item() < min_loss):
            min_loss = loss.data.item()
            print('new min loss!!')
            print('min loss is: ' + str(format(min_loss, '.4f')))
        optimizer.step()
        scheduler.step()
        loss_values.append(loss.data.item())
        if (index > 15 & epoch == 0) or epoch != 0:
            last_10_mean_loss_values.append(np.mean(loss_values[-15:-1]))
    print("epoch "+str(epoch+1)+" loss is: "+str(format(last_10_mean_loss_values[-1],'.4f')))
    loss_per_epoch.append(last_10_mean_loss_values[-1])
plt.plot(last_10_mean_loss_values)
plt.title("training MSE Loss")
plt.xlabel("Iterations")
plt.ylabel("MSE loss")
plt.grid()
plt.show()
plt.plot(loss_per_epoch)
plt.title("training MSE Loss per epoch")
plt.show()


print(orig_image.cpu().data.size())
save_image(noise, './mlp_img/noise.png')
x = to_img(orig_image.cpu().data[0])
save_image(x, './mlp_img/x_1.png')
x_noise = to_img(noised_image.cpu().data[0])
save_image(x_noise, './mlp_img/x_noise_1.png')
x_hat = to_img(output.cpu().data[0])
save_image(x_hat, './mlp_img/x_hat_1.png')

plt.plot(np.array(loss_values), 'r')

torch.save(model.state_dict(), './sim_autoencoder.pth')
torch.save(model, './trained_model.pth')

