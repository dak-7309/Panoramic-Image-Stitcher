# Panoramic Image Stitcher
 
Project Theme and Methodology-


The idea is to create a panorama image stitching system that combines multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image. The process is divided into the following steps: 

Feature Extraction through keypoint detection in images using Harris Corners algorithm
Determine local invariant descriptors in images using methods like SIFT, SURF and other similar methods 
Match corresponding features between images using BruteForce Matcher, that computes the Euclidean distance between features; finding closest features using KNN (K Nearest Neighbours) technique
Homography estimation using RANSAC (RANdom SAmple Consensus), a linear regression ML model robust to outliers; it fits the model to a subset of inliers and returns homography matrix
Applying homography- transform to input images to create the final panorama image, by applying basic operations like rotation, scale, translation, or shear.

Finally as post processing we can plan to use Histogram Equalization to increase global contrast.
