import numpy as np
import cv2

def compute_fft_spectrum(image):
    img_float = np.float32(image)

    dft = np.fft.fft2(img_float)
    
    # shift zero frequency to the center
    dft_shift = np.fft.fftshift(dft)
    
    #calculate Magnitude Spectrum
    magnitude_spectrum = 20 * np.log1p(np.abs(dft_shift))
    
    return dft_shift, magnitude_spectrum