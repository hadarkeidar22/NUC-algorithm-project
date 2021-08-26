from functions import add_no_noises, add_all_noises
import matplotlib.pylab as plt
import numpy as np

def calc_rms(img1,img2):
  diff = img1-img2
  return np.sqrt(np.mean(diff*diff))

def calc_mse(img1,img2):
  diff = img1-img2
  return np.mean(diff*diff)

def WienerFilter(noise_img,filter_blur,sigma=0.1, alpha=0.095):
    # Paste your implementaion of Wiener Filter here
    X,Y = noise_img.shape
    shift_img = np.fft.fftshift(np.fft.fft2(filter_blur,s=(X,Y)))
    x_ax = np.linspace(-X/2,X/2-1, X)
    y_ax = np.linspace(-Y/2,Y/2-1, Y)
    v_ax,u_ax = np.meshgrid(y_ax,x_ax)
    d = np.power(np.abs(shift_img),2) + sigma*sigma*alpha * (np.power(u_ax,2) + np.power(v_ax,2))
    wienner_filter = np.conjugate(shift_img)/d
    denoised_img = np.fft.fftshift(np.fft.fft2(noise_img)) * wienner_filter
    denoised_img = np.abs(np.fft.ifft2(np.fft.ifftshift(denoised_img)))

    return denoised_img

filter_blur = [[0.02104104, 0.0326143,  0.03774462, 0.0326143,  0.02104104],
 [0.0326143 , 0.05055324, 0.0585054 , 0.05055324, 0.0326143 ],
 [0.03774462, 0.0585054,  0.06770845, 0.0585054 , 0.03774462],
 [0.0326143 , 0.05055324, 0.0585054 , 0.05055324 ,0.0326143 ],
 [0.02104104, 0.0326143 , 0.03774462 ,0.0326143,  0.02104104]]
img_dir = r'C:\Users\Hadar Keidar\Desktop\לימודים  ארכיון\שנה ד סמסטר א\פרוייקט\finish_line\small_training'
'''
img = plt.imread(r'C:\Users\Hadar Keidar\Desktop\לימודים  ארכיון\שנה ד סמסטר א\פרוייקט\finish_line\small_training\FLIR_00060.jpeg','rb')
noised_image, noise = add_all_noises(img)
plt.imshow(noised_image, cmap = 'gray')
plt.show()
plt.imshow(img, cmap = 'gray')
plt.show()
loss_values = []
plt.plot(np.array(loss_values))
plt.title("testing MSE Loss")
plt.xlabel("X axis label")
plt.ylabel("Y axis label")
plt.grid()
plt.show()
#noisenew = np.random.normal(0, 1, size=np.shape(img))

'''