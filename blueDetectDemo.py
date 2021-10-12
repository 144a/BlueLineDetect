import sys
import cv2
import numpy as np
import time
import math

# Record Start time for total frame processing
start = time.time()*1000.0


# Define Upper-Lower Bound of threshold for outer edge (GRAY)
threshvalgray = ((0,0,0),(255,255,40))

# Define Upper-Lower Bound of threshold for blue box (BLUE)
# threshvalblue = ((89,95,158),(98,155,200))
threshvalblue = ((110,35,140),(125,65,190))
# threshvalblue = ((50,60,124),(67.5,82,156))
# threshvalblue = ((41.5,124,124),(52,165,159))
# threshvalblue = ((19,165,125),(22.5,206,175))

# Image Resizing function
def imgresize(timg, scale_percent):
	width = int(timg.shape[1] * scale_percent / 100)
	height = int(timg.shape[0] * scale_percent / 100)
	dim = (width, height)

	# resize image
	return cv2.resize(timg, dim, interpolation = cv2.INTER_AREA) 

# CODE USED FOR READING FILE FROM SINGLE IMAGE
# Read File name
file = sys.argv[1]

# CODE USED FOR READING FILE FROM VIDEO FILE
cap = cv2.VideoCapture(0)

# Set Resolution and framerate
# 320x240-------80-90fps
# 640x480-------35-45fps
# 1280x720------14-17fps
# 1920x1080-----4-5fps (Color Issues as well, blue is yellow)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(5, 120)
file_template = sys.argv[2]

print('Reading from file: ',file)


# Show steps
# 1 is true, 0 is false
disp = 0

# Scale for image
scale = 50

# Percent Difference for failure
percentdiffmax = 0.30
percentdiffmin = 0.00

# Font Data
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
thickness = 2

# Process Template (Will be made into a seperate function later)
# Very simple Process, should not be done more than once on code excecution
img_template = imgresize(cv2.imread(file_template, cv2.IMREAD_UNCHANGED), scale)
hsv_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2HSV)
thresh_template = cv2.inRange(hsv_template, threshvalgray[0], threshvalgray[1])
kernel = np.ones((2,2),np.uint8)
ret_template = cv2.dilate(thresh_template,kernel,iterations = 1)
contours_template, hierarchy = cv2.findContours(ret_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_template, contours_template, -1, (0,255,0), 1)
cv2.imshow('Template', img_template)
cv2.waitKey()
x,y,w,h = cv2.boundingRect(contours_template[0])
extern_template = cv2.contourArea(contours_template[0]) / (w * h)




# CODE USED FOR READING FILE FROM SINGLE IMAGE
# Read an image
# img_gray = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# img_blue = cv2.imread(file, cv2.IMREAD_UNCHANGED)
# imgorig = imgresize(cv2.imread(file, cv2.IMREAD_UNCHANGED), 25)

# Track Number of frames
framecount = 0

while cap.isOpened():
	framecount = framecount + 1
	# CODE USED FOR READING FILE FROM VIDEO FILE
	ret, img_gray = cap.read()
	# cv2.imwrite('bluetest.jpg',img_gray)

	# If ret is false, video is on last frame
	if ret == False:
		break
	img_blue = img_gray.copy()
	imgorig = img_gray.copy()

	# Resize the image
	img_gray = imgresize(img_gray, scale)
	img_blue = imgresize(img_blue, scale)
	imgorig = imgresize(imgorig, scale)

	# Show both images
	if disp == 1:
		cv2.imshow('Resized Image', img_gray)

	# Convert to hsv Colorspace
	hsv_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2HSV)
	hsv_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)

	# Threshhold image
	thresh_gray = cv2.inRange(hsv_gray, threshvalgray[0], threshvalgray[1])
	thresh_blue = cv2.inRange(hsv_blue, threshvalblue[0], threshvalblue[1])

	# Display Threshold
	if disp == 1:
		cv2.imshow('Threshold Gray Outline', thresh_gray)
		cv2.imshow('Threshold Blue Box', thresh_blue)

	# Dilate image to help identify edge
	kernel = np.ones((2,2),np.uint8)
	ret_gray = cv2.dilate(thresh_gray,kernel,iterations = 1)
	ret_blue = cv2.dilate(thresh_blue,kernel,iterations = 2) 

	# Display Morphology
	if disp == 1:
		cv2.imshow('Morphology Gray Outline', ret_gray)
		cv2.imshow('Morphology Blue Box', ret_blue)


	# Calculate Contours
	contours_gray, hierarchy = cv2.findContours(ret_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_blue, hierarchy = cv2.findContours(ret_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(imgorig, contours_gray, -1, (0,255,0), 1)
	cv2.drawContours(imgorig, contours_blue, -1, (0,0,255), 1)

	# Calculate and Display Bounding Rectangle for blue line
	if len(contours_blue) > 0:
		# Find the Largest Contour
		maxarea = 0
		maxindex = 0
		for i in range(len(contours_blue)):
			area = cv2.contourArea(contours_blue[i])
			if area > maxarea:
				maxarea = area
				maxindex = i
		x,y,w,h = cv2.boundingRect(contours_blue[maxindex])
		cv2.rectangle(imgorig, (x, y), (x+w, y+h), (255,0,0),2)
		blue_rect = (x,y,w,h)
	else:
		blue_rect = (0,0,0,0)

	# Display Contours
	if disp == 1:
		cv2.imshow('Contours', imgorig)


	# ------The Actual Important Code------
	# Set up for finding edge of jig

	# Best Match Score
	# If there is no singificant match, it will stay 1
	bestMatch = 1.5

	# Match Score
	match = 0

	# Countour Number
	contnum = 0

	# Run through all contours found, and check for a match with the template contour
	for i in range(len(contours_gray)):
		match = cv2.matchShapes(contours_template[0],contours_gray[i],1,0.0)
		if math.isinf(match):
			match = 0
		# print('Match: ', match)
		# cv2.drawContours(imgorig, contours_gray, i, (255,0,0), 1)
		# cv2.imshow('test', imgorig)
		x,y,w,h = cv2.boundingRect(contours_gray[i])
		extern_contour = cv2.contourArea(contours_gray[i]) / (w * h)
		# print('Extern Percentage: ',  abs(extern_contour - extern_template) / extern_template)
		if match < bestMatch and match < 1.5 and abs(extern_contour - extern_template) / extern_template <= 0.5:
			bestMatch = match
			contnum = i
		# cv2.waitKey()

	# Print to see if we have a match
	# print('best match:',  bestMatch)
	if bestMatch == 1.5:
		imgorig = cv2.putText(imgorig, 'CHECK FOCUS AND MAKE SURE THERE ARE NO OBSTRUCTIONS', org, font, 0.6, (255,0,0), 1, cv2.LINE_AA)
		print('MATCH FAILED, CHECK FOCUS AND MAKE SURE THERE ARE NO OBSTRUCTIONS')
	else:
		print(bestMatch)

		# Display the best contour on image
		# cv2.drawContours(imgorig, contours_gray[contnum], -1, (0,255,0), 1)
		x,y,w,h = cv2.boundingRect(contours_gray[contnum])
		cv2.rectangle(imgorig, (x, y), (x+w, y+h), (0,0,255),2)
		gray_rect = (x,y,w,h)

		# Check for pass/fail
		diff = blue_rect[1] - gray_rect[1]
		print(diff/gray_rect[3])

		if diff < 0 or not(diff/gray_rect[3] > percentdiffmin and diff/gray_rect[3] < percentdiffmax):
			imgorig = cv2.putText(imgorig, 'FAIL------FAIL------FAIL', org, font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
			print('-----FAIL-----')
		else:
			imgorig = cv2.putText(imgorig, 'PASS++++++PASS++++++PASS', org, font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
			print('+++++PASS+++++')

	cv2.imshow('Final Contour', imgorig)

	# Give small delay for image to be shown on screen and exit when 'q' is pressed
	ch = cv2.waitKey(1)
	if ch == ord('q'):
	        break


# Print total time for frame
print('total time: ', time.time()*1000.0-start)

print('Average FPS', framecount/((time.time()*1000.0-start)/1000))

cap.release()
cv2.destroyAllWindows()
