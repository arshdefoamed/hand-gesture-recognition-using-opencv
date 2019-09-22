*This project is under construction.*
This project is submitted as part of evaluation component of a computer vision course. This project follows the guidelines by the following publication:
Hand tracking and gesture recognition system for human-computer interaction using low-cost hardware by Hui-Shyong Yeo & Byung-Gook Lee & Hyotaek Lim 

Following are the steps followed:
1. Retrieve Image
2. Change color space
3. Extract binary image 
4. Apply Morphology
5. Extract the area of largest contour
By now you should have hand region alongwith the arm
6. Find maximum insrcibed circle of the extracted area
7. Limit the region of interest using center and radius of maximum insribed circle
8. Apply steps 5-7 again on region of interest
9. Find convex hull and convexity defects and count the number of fingers

Following two applications have been developed :
1. Counting number of fingers shown by the hand
2. Right swipe and left swipe detection

Future Work:
1. Background subtraction
2. Face removal before processing
3. Applying convolutional neural networks

If this repository is helpful to you give it a star :)
