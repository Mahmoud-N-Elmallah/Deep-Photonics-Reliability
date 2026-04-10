import numpy as np
import cv2

def compute_fft_spectrum(image):
    img_float = np.float32(image)
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log1p(np.abs(dft_shift))
    return dft_shift, magnitude_spectrum

def apply_gaussian_notch_filter(dft_shift, width=0.9, notch_depth=0.99):
    r, c = dft_shift.shape
    cr, cc = r // 2, c // 2

    x = np.arange(-cc, cc)
    y = np.arange(-cr, cr)

    gauss_v = np.exp(-(x**2) / (2 * width**2))
    gauss_h = np.exp(-(y**2) / (2 * width**2))

    mask_v = 1 - notch_depth * gauss_v
    mask_h = 1 - notch_depth * gauss_h
    mask = np.minimum(mask_v[np.newaxis, :], mask_h[:, np.newaxis])

    #protects the DC component
    center_radius = 6
    mask[cr-center_radius:cr+center_radius, cc-center_radius:cc+center_radius] = 1

    return dft_shift * mask, mask

def reconstruct_image(filtered_dft, apply_bilateral=True):
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    recon = np.abs(img_back)
    
    #
    # Scales results back to a viewable 0-255 range
    recon_rescaled = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if apply_bilateral:
        recon_rescaled = cv2.bilateralFilter(recon_rescaled, d=7, sigmaColor=50, sigmaSpace=50)
        
    return recon_rescaled

def enhance_defects_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    # Applies CLAHE to heavily contrast the faults against the background
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def enhance_defects_sobel(image):
    # Applies Sobel edge detection to highlight cracks
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
