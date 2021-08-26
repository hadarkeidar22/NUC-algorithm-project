# NUC-algorithm-project
Position-dependent CNN aproach calibration for bolometric infrared camera's images. The project was crafted with python and the use of NumPy, matplotlib and PyTorch modules.

First, to create the training dataset for the CNN a FPN(fixed patterned noise) simulation module was implemented using NumPy package, when the location of the noised columns is determined by the complex column FPN desired pattern's function which is properly randomized, and the noise multiplication parameter is determined to be linear. For a right use of the desired model,  pairs of images had to be inserted as an input to the CNN. The pairs should consist a noised image and the same image but noise clean. The FPN simulation module takes a clean infra red image and simulates the desired noise (FPN, as a low-cost camera generates which is influenced by its low bolometric differentiation and more salt & pepper and gaussian by location noise).

An implementation of the CNN design was schemed as a 3 layers convolution encoder and a reverse 3 layers convolutional decoder, the non-linearity activation functions was determined to be ReLU and Tanh:

![image](https://user-images.githubusercontent.com/72237098/130953685-007d2511-210e-4985-9b99-e246036fbefa.png)

