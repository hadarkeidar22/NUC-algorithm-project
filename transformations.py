from torchvision import transforms
from functions import add_all_noises

#augmentation methods

myTransforms = transforms.Compose([
  transforms.Lambda(add_all_noises),                        #add_all_noises(image)
  transforms.ToPILImage(),
  #ColorJitter(brightness = (0.6, 0.9)),          #(min,max) = (0.6, 0.9)
  #transforms.RandomRotation(degrees = (-20, 20)),           #(min,max) = (-20, 20) #should be back
  transforms.RandomHorizontalFlip(p=0),
  transforms.ToTensor()
])