import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.segmentation import active_contour

def im2double(im):
    return im.astype(np.float64) / np.iinfo(im.dtype).max

def daugman_operator(image, rmin, rmax):
    rows, cols = image.shape
    max_gradient = 0
    best_center = (0, 0)
    best_radius = 0
    
    for x in range(rmin, rows - rmin):
        for y in range(rmin, cols - rmin):
            for r in range(rmin, rmax):
                mask = np.zeros_like(image)
                cv2.circle(mask, (y, x), r, 1, thickness=1)
                gradient = np.sum(gaussian_gradient_magnitude(image * mask, sigma=1))
                if gradient > max_gradient:
                    max_gradient = gradient
                    best_center = (x, y)
                    best_radius = r
    
    return best_center, best_radius

def drawcircle(image, center, radius):
    output = image.copy()
    cv2.circle(output, (int(center[1]), int(center[0])), int(radius), (255, 0, 0), 2)
    return output

def apply_active_contour(image, init_snake):
    snake = active_contour(image, init_snake, alpha=0.01, beta=0.1, gamma=0.1)
    return snake

def apply_dfs(image):
    rows, cols = image.shape
    transformed = np.zeros_like(image, dtype=np.complex128)
    
    for u in range(rows):
        for v in range(cols):
            sum_value = 0
            for x in range(rows):
                for y in range(cols):
                    sum_value += image[x, y] * np.exp(-2j * np.pi * ((u * x / rows) + (v * y / cols)))
            transformed[u, v] = sum_value
    
    magnitude_spectrum = 20 * np.log(np.abs(transformed) + 1)
    return magnitude_spectrum

def irisSeg(filename, rmin, rmax, view_output=False):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = im2double(image)
    image = erosion(image)
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Initial Grayscale Image')
    plt.show()
    
    # Applying Daugman’s integrodifferential operator
    (x, y), pupil_radius = daugman_operator(image, rmin, rmax)
    iris_radius = pupil_radius * 2  # Approximation
    
    segmented_img = drawcircle(image, (x, y), iris_radius)
    segmented_img = drawcircle(segmented_img, (x, y), pupil_radius)
    
    plt.figure()
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Daugman Operator - Detected Boundaries')
    plt.show()
    
    # Applying Active Contour Model
    s = np.linspace(0, 2 * np.pi, 100)
    init_snake = np.array([x + iris_radius * np.cos(s), y + iris_radius * np.sin(s)]).T
    snake = apply_active_contour(image, init_snake)
    
    plt.figure()
    plt.imshow(segmented_img, cmap='gray')
    plt.plot(snake[:, 1], snake[:, 0], '-r', lw=2)
    plt.title('Active Contour Segmentation')
    plt.show()
    
    # Applying DFS
    magnitude_spectrum = apply_dfs(image)
    
    plt.figure()
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('DFS Magnitude Spectrum')
    plt.show()
    
    return (x, y, iris_radius), (x, y, pupil_radius), segmented_img

if __name__ == '__main__':
    coord_iris, coord_pupil, output_image = irisSeg('eye.png', 40, 70, view_output=True)
    print("Iris Coordinates:", coord_iris)
    print("Pupil Coordinates:", coord_pupil)
