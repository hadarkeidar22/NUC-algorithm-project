from torch.utils.data import Dataset
import matplotlib.pylab as plt
import os
import numpy as np
from functions import add_no_noises, add_all_noises
from torchvision.transforms import ToTensor, RandomHorizontalFlip, ToPILImage, RandomRotation, RandomAffine




SAMPLE = 'sample'
LABEL = 'label'
FPN = 'FPN'

class MyDataset(Dataset):

    def __init__(self, img_dir='', transforms=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.img_names.sort()
        self.img_names = [os.path.join(img_dir, img_names) for img_names in self.img_names]
        self.transforms = transforms

    def __getitem__(self, index):
        img_name = self.img_names[index]
        orig_image = plt.imread(img_name)
        if self.transforms:
            orig_image = ToPILImage()(orig_image)
            #Random Transformations
            if np.random.normal(0.5) > 0.5:
                orig_image = RandomHorizontalFlip(p=1)(orig_image)
            if np.random.normal(0.5) > 0.5:
                orig_image = RandomAffine(degrees=10)(orig_image)
            sample, all_noises = add_all_noises(orig_image)
            sample = ToTensor()(sample)
            label = add_no_noises(orig_image)
            label = ToTensor()(label)
        return {SAMPLE: sample, LABEL: label, FPN: all_noises}

    def __len__(self):
        return len(self.img_names)

