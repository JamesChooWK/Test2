# import the picture
import numpy as np
import cv2

# Read image
img = cv2.imread('1original.png')
# Show image
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original', 640, 480)
cv2.imshow('original', img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert color space to hsv
#  There are a couple of informative images about HSV in the draft doc

# HSV COLOR SPACE
# This is red in HSV (need to split into two)
min_colour = (140, 20, 100)
max_colour = (180, 255, 255)
min_colour2 = (0, 20, 100)
max_colour2 = (20, 255, 255)
min_colour_hsv = cv2.cvtColor(np.uint8([[min_colour]]), cv2.COLOR_BGR2HSV)[0][0]
max_colour_hsv = cv2.cvtColor(np.uint8([[max_colour]]), cv2.COLOR_BGR2HSV)[0][0]

# Simple thresholding, you can also make a mask with this but it is just a > criterion instead of the
# < and > that of inRange()
# ret, thresh = cv2.threshold(hsv, 127, 255, 0)

red_mask1 = cv2.inRange(hsv, tuple(min_colour), tuple(max_colour))
red_mask2 = cv2.inRange(hsv, tuple(min_colour2), tuple(max_colour2))
# All pixels in range of min and max colours are now white and the rest black, combine the two masks
red_mask = red_mask1 + red_mask2

# Show mask
# cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('mask', 640, 480)
# cv2.imshow('mask', red_mask)

# Now two detection strategies: contours and blobs

# Find the contours of the white blobs.
img2, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
# Draw the bounding rectangle (in green)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
img11 = cv2.imread('1original.png')
cv2.drawContours(img, c, -1, (255,0, 0), 1)
# cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('contours', 640, 480)
# cv2.imshow('contours',img)


## cropping the contour out
amp_img = img11[y:y+h,x:x+w]

cv2.namedWindow('amplify1', cv2.WINDOW_NORMAL)
cv2.imshow('amplify1', amp_img)
amp_img = cv2.cvtColor(amp_img,cv2.COLOR_RGB2GRAY)


amp_img = cv2.GaussianBlur(amp_img,(3,3),3)
amp_img = cv2.Laplacian(amp_img,cv2.CV_64F)
# amp_img = cv2.GaussianBlur(amp_img,(5,5),2)
kernel = np.ones((3,3),np.uint8)

# amp_img=cv2.resize(amp_img,dsize=(40,40),fx=5,fy=5) ## added this on 26/1
# amp_img = cv2.dilate(amp_img,kernel,iterations =1)
# amp_img = cv2.erode(amp_img,kernel,iterations=1)
amp_img = cv2.morphologyEx(amp_img, cv2.MORPH_CLOSE, kernel)

#rotates the picture
rows,cols = amp_img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
amp_img = cv2.warpAffine(amp_img,M,(rows,cols))

# cv2.imwrite('binary_image.png', cv2.resize(amp_img,dsize=(40,40),fx=0,fy=0)*255)
cv2.imwrite('binary_image.png', amp_img*255,)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img',amp_img*255)

# #set up filter to select white colour background
gg = cv2.imread('binary_image.png' )
# gg = cv2.imread('image_61.jpg')
white_up = (255,255,255)
white_low = (250,250,250)
white_mask = cv2.inRange(gg,white_low,white_up)

# Find the contours of the white blobs.
_, contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
print(x,y,w,h) # checking

# cv2.drawContours(gg, c, -1, (255,0, 0), 5)
cv2.namedWindow('white', cv2.WINDOW_NORMAL)
cv2.imshow('white',gg[y:y+h,x:x+w])

cv2.imwrite('test1.png',gg[y:y+h,x:x+w])

#---------------------------google tesseract(character recognition)---------------------------------------------------##
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" #need this to run
# feb_elec_text = pytesseract.image_to_string(amp_img[y:y+h,x:x+w],config="--psm 10" )
feb_elec_text = pytesseract.image_to_string(amp_img[y:y+h,x:x+w],lang='eng', config="--psm 10 --oem 1")
print('The detected character is '+feb_elec_text)
# pytesseract



while True:
    keyPressed = cv2.waitKey(1)
    if keyPressed == 27:
        cv2.destroyAllWindows()
        break