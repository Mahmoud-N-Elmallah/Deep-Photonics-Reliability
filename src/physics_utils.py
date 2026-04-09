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


def apply_notch_filter(dft_shift, centers, radius=5):
    r, c = dft_shift.shape
    mask = np.ones((r, c))
    
    for (x,y) in centers:
        cv2.circle(mask, (x, y), radius, 0, -1)
 
    filtered_dft = dft_shift * mask
    return filtered_dft, mask



def reconstruct_image(filtered_dft):
    
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)



def apply_gaussian_notch_filter(dft_shift,width=5,notch_depth=0.95):
    """
    new filter insteead of the earlier one
     Narrow suppression around axes
     Preserves low frequencies (DC region)
     Avoids over-smoothing
    """

    r, c = dft_shift.shape
    cr, cc = r // 2, c // 2

    x = np.arange(-cc, cc)
    y = np.arange(-cr, cr)

    # Narrow Gaussian dips (not full suppression)
    gauss_v = np.exp(-(x**2) / (2 * width**2))
    gauss_h = np.exp(-(y**2) / (2 * width**2))

    # Convert to "notch" (1 = keep, <1 = suppress)
    mask_v = 1 - notch_depth * gauss_v
    mask_h = 1 - notch_depth * gauss_h

    # Build 2D mask \wo overkilling diagonals
    mask = np.minimum(mask_v[np.newaxis, :], mask_h[:, np.newaxis])

    # Preserve central low frequencies this is the part that was missing in the first version of this filter
    center_radius = 6
    mask[cr-center_radius:cr+center_radius,
         cc-center_radius:cc+center_radius] = 1

    return dft_shift * mask, mask

