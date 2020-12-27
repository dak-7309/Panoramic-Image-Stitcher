#!/usr/bin/python

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
import time

cv2.ocl.setUseOpenCL(False)
imgA =str(sys.argv[1])
imgB =str(sys.argv[2])

img1=imageio.imread(imgA)
img2=imageio.imread(imgB)
# Read images and converting them to gray scale images for computation
img2_gray=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img1_gray=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

# arbitary variables for sizing and plotting graphs
ROW=1
COL=2
font_=15

# Plotting graphs 1 X 2 subplot formating 
fig, (A1, A2)=plt.subplots(nrows=ROW, ncols=COL,figsize=(20,8))
A1.imshow(img2, cmap="gray")
A1.set_xlabel("First Image", fontsize=font_)
A2.imshow(img1, cmap="gray")
A2.set_xlabel("Second Image",fontsize=15)
#saving the file to another file for displaying purposes
plt.savefig("sift_knn:_original_images_1_and_2.jpg")
plt.show()

#creating an sift object, followed by detecting and extracting relevant features from our given images
#  (keypointsA, featuresA) contains these these features
t_sift_knn = time.time()
(keypointsA, featuresA) = cv2.xfeatures2d.SIFT_create().detectAndCompute(img1_gray, None)
(keypointsB, featuresB) = cv2.xfeatures2d.SIFT_create().detectAndCompute(img2_gray, None)

def plot_features(keypointsA,keypointsB,img1_gray,img2_gray):
    fig, (A1,A2) = plt.subplots(nrows=ROW, ncols=COL, figsize=(20,8))
    # plotting features on the first image itself
    A1.imshow(cv2.drawKeypoints(img1_gray,keypointsA,None,color=(255,0,0)))
    A1.set_xlabel("figA", fontsize=font_)
     # plotting features on the first image itself
    A2.imshow(cv2.drawKeypoints(img2_gray,keypointsB,None,color=(255,0,0)))
    A2.set_xlabel("figB", fontsize=font_)
    # saving them to another jpg file
    plt.savefig("sift_bf:features.jpg")
    plt.show()

# calling the plotting function from above
plot_features(keypointsA,keypointsB,img1_gray,img2_gray)
fig = plt.figure(figsize=(20,8))

# Creating a matcher object for hamming distances 

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# Matcher only returns the matches with indexing (i,j) if the i'th index of the first descriptor matches 
# the j'th decriptor of the second
best_matches = bf.match(featuresA,featuresB)
# sorting the data according to their distances 
intmatch = sorted(best_matches, key = lambda x:x.distance)


# drawMatches maps the the matches obtained to identify the point of overlap
img3 = cv2.drawMatches(img1,keypointsA,img2,keypointsB,intmatch[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plotting and saving the above results in a new file
plt.imshow(img3)
plt.savefig("sift_bf:matches.jpg")
plt.show()



#  finding the trasformation matrix for the second image
def homography(keypointsA, keypointsB, featuresA, featuresB, matches, reprojThresh):
    
    a=[]
    for key in keypointsA:
        a.append(key.pt)
    A=np.array(a)

    b=[]
    for key in keypointsB:
        b.append(key.pt)
    B=np.array(b)

    if len(matches)>=3:

        a=[]
        b=[]

        for m in matches:
            a.append(A[m.queryIdx])
            b.append(B[m.trainIdx])
        
        ptsA=np.float32(a)
        ptsB=np.float32(b)
        # using ransac algorithm we determine 3 X 3 transformation matrix 
        # to bring both the images to same perspective 
        (H,status)=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)

        return (matches, H, status)
    else:
        return None

# calling homography method
(matches,H,status)=homography(keypointsA, keypointsB, featuresA, featuresB, intmatch, reprojThresh=4)
print("Homography= \n",H)

R=img2.shape[0]
C=img2.shape[1]

# apply perspective transformation onto the first image using the obtained homography 
# and warping it with the other image
finale=cv2.warpPerspective(img1,H,(img1.shape[1]+img2.shape[1],img1.shape[0]+img2.shape[0]))
finale[0:R,0:C]=img2


def transform_threshold(finale):
    # convert the processed grayscale image back to RGB image
    colour=cv2.cvtColor(finale, cv2.COLOR_BGR2GRAY)
    # limitting it to 0-255 range
    colourthresh=cv2.threshold(colour, 0, 255, cv2.THRESH_BINARY)[1]
 # Noramlizing the results obtained
    colourthresh = cv2.equalizeHist(colourthresh)
    #  finding the contours in the image, it would help to 
    contouring=cv2.findContours(colourthresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contouring=imutils.grab_contours(contouring)


    MAX=max(contouring,key=cv2.contourArea)

    # finding the 4-coordinates and dimensions of the bounding rectangle for the final image
    (x_cood,y_cood,width,height)=cv2.boundingRect(MAX)
    # cropping the finalimage out of the finale using the dimenstion obtained from cv2.boundingRect

    final=finale[y_cood:y_cood+height, x_cood:x_cood + width]


    plt.figure(figsize=(20,8))
    plt.imshow(final)
    plt.savefig("sift_bf:final_transformed_image.jpg")



# calling the transform function to the final 
transform_threshold(finale)
t_sift_bf_1 = time.time()
