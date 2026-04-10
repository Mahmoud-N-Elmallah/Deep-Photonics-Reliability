import numpy as np
import cv2

def compute_fft_spectrum(image):
    img_float = np.float32(image)
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log1p(np.abs(dft_shift))
    return dft_shift, magnitude_spectrum

def apply_gaussian_notch_filter(dft_shift, width=5, notch_depth=0.95):
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




#new enhanced filter

def adaptive_notch_filter(dft_shift, width=0.05, depth=0.99):
    
    mag = np.abs(dft_shift)
    r, c = mag.shape
    cr, cc = r // 2, c // 2
    y, x = np.indices((r, c))
    x = x - cc
    y = y - cr
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    mag = mag * (radius > 10)
    # histogram over angles
    num_bins = 180
    hist, bins = np.histogram(theta, bins=num_bins, weights=mag)
    # smooth histogram
    hist = np.convolve(hist, np.ones(5)/5, mode='same')
    # find top 2 peaks
    peak_idxs = np.argsort(hist)[-2:]
    dominant_angles = bins[peak_idxs]
    # build mask
    mask = np.ones((r, c))
    for angle in dominant_angles:
        angle_diff = np.angle(np.exp(1j * (theta - angle)))
        gauss = np.exp(-(angle_diff**2) / (2 * width**2))
        mask *= (1 - depth * gauss)
    mask[cr-6:cr+6, cc-6:cc+6] = 1

    return dft_shift * mask, mask