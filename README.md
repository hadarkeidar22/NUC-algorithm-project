# NUC-algorithm-project
Position-dependent CNN aproach calibration for bolometric infrared camera's images. The project was crafted with python and the use of NumPy, matplotlib and PyTorch modules.

First, to create the training dataset for the CNN a FPN(fixed patterned noise) simulation module was implemented using NumPy package, when the location of the noised columns is determined by the complex column FPN desired pattern's function which is properly randomized, and the noise multiplication parameter is determined to be linear. For a right use of the desired model,  pairs of images had to be inserted as an input to the CNN. The pairs should consist a noised image and the same image but noise clean. The FPN simulation module takes a clean infra red image and simulates the desired noise (FPN, as a low-cost camera generates which is influenced by its low bolometric differentiation and more salt & pepper and gaussian by location noise).

![image](https://user-images.githubusercontent.com/72237098/130953086-5d3bbb1a-02cc-4124-bb3d-6de4a22371ac.png)
