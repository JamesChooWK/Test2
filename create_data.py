# Run this code as command line code, main input: -r, -num, - o
import argparse
import cv2
import os
import numpy

# to rotate the image
def rotation(img,angle,num,arg):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    rows, cols = img.shape
    centre = (cols/2,rows/2)
    path = 'F:\GDP\G'
    if arg:
        for i in range(num):
            M = cv2.getRotationMatrix2D(centre,angle*i,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            cv2.imwrite(os.path.join(path,'image_'+str(i)+'.jpg'),dst)

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rotation", help="random rotation of image from 0 to rotation angle", type=int, default=0)
    parser.add_argument("-num","--numItr", help="Number of Iteration", type =int)
    parser.add_argument("-o","--output",help="output the result to a file", action = "store_true")
    args = parser.parse_args()

    rall_img = cv2.imread('binary_image.png')
    rotation(rall_img,args.rotation,args.numItr,args.output)



if __name__ == '__main__':
    Main()
    print('The process is done.')
