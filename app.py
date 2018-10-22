from PIL import Image, ImageFilter
import cv2
import numpy as np
import argparse

'''
def check_pos(img, pos):
    if pos > 0:
        return img[pos]
    return None


img = Image.open('drawn_venom_face.jpg')

grey_scaled_img = img.filter(ImageFilter.FIND_EDGES).convert('L')

grey_scale_check = lambda i : i > 120

new_img_data = []
grey_scaled_img_data = grey_scaled_img.getdata()
grey_scaled_img_size = grey_scaled_img.size

pos = 0
surrounding_pos = []

for intensity in grey_scaled_img_data:
    if grey_scale_check(intensity):
        # if ( check_pos(grey_scaled_img_data,pos - 1) or
        #         check_pos(grey_scaled_img_data,pos + 1) or
        #         check_pos(grey_scaled_img_data,pos - grey_scaled_img_size[0]-1) or
        #         check_pos(grey_scaled_img_data,pos - grey_scaled_img_size[0]) or
        #         check_pos(grey_scaled_img_data,pos - grey_scaled_img_size[0]+1) or
        #         check_pos(grey_scaled_img_data,pos + grey_scaled_img_size[0]) or
        #         check_pos(grey_scaled_img_data,pos + grey_scaled_img_size[0]-1) or
        #         check_pos(grey_scaled_img_data,pos + grey_scaled_img_size[0]+1) ):
        new_img_data.append((255, 255, 255))

        # try:
        #     surrounding_pos.append([pos, check_pos(grey_scaled_img_data,pos - grey_scaled_img_size[0]-1),
        #                                 check_pos(grey_scaled_img_data, pos - grey_scaled_img_size[0]),
        #                                 check_pos(grey_scaled_img_data, pos - grey_scaled_img_size[0] + 1),
        #                                 check_pos(grey_scaled_img_data,pos - 1),
        #                                 check_pos(grey_scaled_img_data,pos + 1),
        #                                 check_pos(grey_scaled_img_data, pos + grey_scaled_img_size[0] - 1),
        #                                 check_pos(grey_scaled_img_data,pos + grey_scaled_img_size[0]),
        #                                 check_pos(grey_scaled_img_data,pos + grey_scaled_img_size[0]+1)])
        # except:
        #         pass

    else:
        new_img_data.append((0,0,0))

    pos += 1

new_img = Image.new(img.mode, img.size)
new_img.putdata(new_img_data)
new_img.save('greyscale1.png')
'''
import cv2;
import numpy as np;

# Read image
im_in = cv2.imread("drawn_venom_face.jpg", cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);

cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
	# draw the contour and show it
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

# Display images.
cv2.imshow("Thresholded Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)

'''
input_image = cv2.imread('greyscale1.png', 0)
input_image = cv2.threshold(input_image, 1, 255, cv2.THRESH_BINARY)[1]
input_image_comp = cv2.bitwise_not(input_image) #invert image


kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], np.uint8)
kernel2 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], np.uint8)


kernel = np.ones((5,5), np.uint8)

opened_image = cv2.morphologyEx(input_image_comp, cv2.MORPH_OPEN, kernel)
dilated_image = cv2.dilate(opened_image, kernel, iterations=1)

#hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel1) # cv2.MORPH_ERODE
#hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_OPEN, kernel2)
#hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)

cv2.imwrite('isolated1.png', opened_image)
'''