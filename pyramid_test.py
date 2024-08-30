import cv2
from complex_steerable_pyramid import *

image = cv2.imread("examples/mona.png", cv2.IMREAD_GRAYSCALE)
assert image is not None, 'cannot open the file'
image = cv2.resize(image, None, fx=0.5, fy=0.5)


depth = 3
orientations = 6
pyramid, filters = complex_steerable_pyramid(image, depth, orientations)
recon_image = reconstruction(pyramid, filters)

print(f'imagesize:{image.shape}, reconsize:{recon_image.shape}')
plt.subplot(1,2,1), plt.imshow(image)
plt.subplot(1,2,2), plt.imshow(recon_image)
plt.show()

pyramid_display(pyramid, depth, orientations)
assert np.sum(np.abs(recon_image - image)) < 1, 'error is too much'
