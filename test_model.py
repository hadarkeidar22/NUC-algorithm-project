from setups import MyDataset
from transformations import myTransforms
import numpy as np
from matplotlib import pylab as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from functions import add_no_noises
from torchvision.utils import save_image
from my_model import Autoencoder

def to_img(x):
    x = x.view(x.size(0), 1, 512, 640)
    return x

SAMPLE = 'sample'
LABEL = 'label'
FPN = 'FPN'

batch_size = 1
img_dir = r'C:\Users\Hadar Keidar\Desktop\לימודים  ארכיון\שנה ד סמסטר א\פרוייקט\finish_line\small_test'
noise_dic = r'C:\Users\Hadar Keidar\Desktop\לימודים  ארכיון\שנה ד סמסטר א\פרוייקט\finish_line\noise.png'
testset = MyDataset(img_dir=img_dir, transforms=myTransforms)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

trained_model = Autoencoder()
trained_model.load_state_dict(torch.load('./sim_autoencoder.pth'))
trained_model.eval()
criterion = nn.MSELoss()
testset_size = len(testset)
loss_values = []
last_10_mean_loss_values = []
loss_per_epoch = []
plt.plot(np.array(loss_values), 'r')
min_loss = 1
this_noise = plt.imread(noise_dic)
this_noise = add_no_noises(this_noise[:,:,0])
index = 1
#save_image(this_noise, './img_4_report/noise.png')
with torch.no_grad():
    for data in testloader:
        orig_image = data[LABEL]
        noised_image = np.add(orig_image[0][0], this_noise)
        noised_image = np.clip(noised_image, 0, 1)
        print('in process: ' + str(format((batch_size * 100 * (index / testset_size)), '.3f')) + "%")
        index = index + 1
        output = trained_model(orig_image)
        loss = criterion(output, orig_image)
        print('min loss is: ' + str(format(loss.data.item(), '.5f')))
        loss_values.append(loss.data.item())
        if (index > 15):
            last_10_mean_loss_values.append(np.mean(loss_values[-13:-1]))
        '''x = to_img(orig_image.cpu().data[0])
        save_image(x, './img_4_report/x_orig'+str(index)+'.png')
        #x_noise = to_img(noised_image[0])
        save_image(noised_image, './img_4_report/x_noised_'+str(index)+'.png')
        x_hat = to_img(output.cpu().data[0])
        save_image(x_hat, './img_4_report/x_hat_'+str(index)+'.png')'''

plt.plot(last_10_mean_loss_values)
plt.title("testing MSE Loss [gs]")
plt.xlabel("test iterations")
plt.ylabel("MSE loss")
plt.grid()
plt.show()
print('mean loss value: '+str(np.mean(loss_values)))
print('std loss value: '+str(np.std(loss_values)))


