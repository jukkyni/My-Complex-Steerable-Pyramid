import cv2
import matplotlib.pyplot as plt
from complex_steerable_pyramid import *

image = cv2.imread("examples/demo02.jpg", cv2.IMREAD_COLOR)
assert image is not None, 'cannot open the file'
image = cv2.resize(image, None, fx=0.5, fy=0.4)

B_img, G_img, R_img = cv2.split(image)

depth = 6
orientations = 4
pyramid, filters = complex_steerable_pyramid(image, depth, orientations)
recon_image = reconstruction(pyramid, filters)

recon_image = np.round(recon_image).astype(np.uint8)

print(f'image:size={image.shape}, dtype={image.dtype}, recon:size={recon_image.shape}, dtype={recon_image.dtype}')
# pyramid_display(pyramid[:,:,:,0], depth, orientations)
assert np.sum(np.abs(recon_image - image)) < 1, f'error=({np.sum(np.abs(recon_image - image))}) is too much'

plt.subplot(1,2,1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(recon_image, cv2.COLOR_BGR2RGB))
plt.show()