# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import pytesseract
from PIL import Image
from pytesseract import Output
import pandas as pd
import textwrap

def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def get_string(image):
	thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	thresh = cv2.GaussianBlur(thresh, (3,3), 0)
	data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 1')
	dedented_text = textwrap.dedent(data).strip()
	return data
	
'''	
def get_string(image):

	#thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	#thresh = cv2.GaussianBlur(thresh, (3,3), 0)
	custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1 -l eng+ita'
	d = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
	df = pd.DataFrame(d)

	# clean up blanks
	df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
	# sort blocks vertically
	sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
	for block in sorted_blocks:
		curr = df1[df1['block_num']==block]
		sel = curr[curr.text.str.len()>3]
		char_w = (sel.width/sel.text.str.len()).mean()
		prev_par, prev_line, prev_left = 0, 0, 0
		text = ''
		for ix, ln in curr.iterrows():
			# add new line when necessary
			if prev_par != ln['par_num']:
				text += '\n'
				prev_par = ln['par_num']
				prev_line = ln['line_num']
				prev_left = 0
			elif prev_line != ln['line_num']:
				text += '\n'
				prev_line = ln['line_num']
				prev_left = 0

			added = 0  # num of spaces that should be added
			if ln['left']/char_w > prev_left + 1:
				added = int((ln['left'])/char_w) - prev_left
				text += ' ' * added
			text += ln['text'] + ' '
			prev_left += len(ln['text']) + added + 1
		text += '\n'
		print(text)
		return text
'''
# construct the argument parser and parse the arguments
argParse = argparse.ArgumentParser()
argParse.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
argu = vars(argParse.parse_args())
file = open("text_file.txt","w")
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(argu["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("image", image)
cv2.imshow("Edged", edged)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
		
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.imwrite("result.jpg", warped)
width = int(warped.shape[1] * 200 / 100)
height = int(warped.shape[0] * 200 / 100)
dsize = (width, height)
warped = cv2.resize(warped, dsize)
file.write(get_string(warped))
file.close()
cv2.waitKey()
cv2.destroyAllWindows()
