# python mosaic.py inputImage.jpg imageDatasetPath outputImage.png
#cat list.txt | awk '{printf("%s%s%s%s%s\n", "python3 mosaic.py ~/mosaic/", $1, " ~/myImages/ ~/mosaicOut/", $1, "_512.jpg color 512");}'
import os, sys, cv2, math, re
import numpy as np
import pickle

scales = 4
step = int(sys.argv[5]) #Image patch size
tbSize = 32
minError = 64
maxOutSize = 8196

if sys.argv[4] == 'gray' or not sys.argv[4] == 'color':
	gray = True # process the image in RGB or gray
else:
	gray = False

def covert2gray(img):
	tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(0,3):
		img[:,:,i] = tmp

	return img

def getCenterSquare(tmp):
	minSize = np.min(tmp.shape[0:2])
	if tmp.shape[0] == minSize:
		shift = int((tmp.shape[1] - minSize)/2)
		tmp = tmp[:,shift:shift+tmp.shape[0],:]
	else:
		shift = int((tmp.shape[0] - minSize)/2)
		tmp = tmp[shift:shift+tmp.shape[1],:,:]
	
	return tmp
	
def smoothBorders(result):
	for i in range(step, result.shape[0]-step, step):
		avg = result[i,:,:]*0.5 + result[i-1,:,:]*0.5
		result[i,:,:] = result[i,:,:]*0.5 + avg*0.5
		result[i-1,:,:] = result[i-1,:,:]*0.5 + avg*0.5
	for i in range(step, result.shape[1]-step, step):
		avg = result[:,i,:]*0.5 + result[:,i-1,:]*0.5
		result[:,i,:] = result[:,i,:]*0.5 + avg*0.5
		result[:,i-1,:] = result[:,i-1,:]*0.5 + avg*0.5
			
	return result

def computeLayer(img, thumbs, k, result, distMap, factor):
	stepf = step * factor
	for i in range(0, img.shape[0]-stepf-1, stepf):
		for j in range(0, img.shape[1]-stepf-1, stepf):
			patch = img[i:i+stepf,j:j+stepf,:]
			patch = cv2.resize(patch, (tbSize, tbSize))
			# TODO Convert dist to a matrix (more precise)
			dist = 256
			index = 0
			for m in range(0, k):
				dist_tmp = np.sum(np.abs(patch-thumbs[m,:,:,:]))/(tbSize * tbSize * 3)
				if dist_tmp < dist:
					dist = dist_tmp
					index = m

			tmp = cv2.imread(sys.argv[2]+filenames[index], cv2.IMREAD_COLOR)
			if gray:
				tmp = covert2gray(tmp)
			tmp = getCenterSquare(tmp)
			if step == 1:
				cv2.medianBlur(tmp, 5)
			else:
				cv2.medianBlur(tmp, 3)
			tmp = cv2.resize(tmp, (stepf, stepf))

			if (dist < np.mean(distMap[i:i+stepf,j:j+stepf])):
				distMap[i:i+stepf,j:j+stepf] = dist
				if dist < minError:
					result[i:i+stepf,j:j+stepf,:] = tmp
				else:
					result[i:i+stepf,j:j+stepf,:] = img[i:i+stepf,j:j+stepf,:]

	if factor == 1:
		result = smoothBorders(result)

	return result, distMap

print(sys.argv[1])

# img = reference image
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
if gray:
	img = covert2gray(img)
times = int(round(maxOutSize/np.max(img.shape[0:2])))
img = cv2.resize(img, (int(img.shape[1]*times), int(img.shape[0]*times)))

result = np.zeros((img.shape[0], img.shape[1], 3), int)
distMap = np.ones((img.shape[0], img.shape[1]), float)*256

filenames = os.listdir(sys.argv[2])
if not os.path.isfile(sys.argv[4]+'.pkl'):
	thumbs = np.zeros((len(filenames), tbSize, tbSize, 3), float)
	print('Rading images')
	for i in range (0, len(filenames)):
		tmp = cv2.imread(sys.argv[2]+filenames[i], cv2.IMREAD_COLOR)
		if gray:
			tmp = covert2gray(tmp)
		tmp = getCenterSquare(tmp)
		tmp = cv2.medianBlur(tmp, 11)
		tmp = cv2.resize(tmp, (tbSize, tbSize))
		thumbs[i,:,:,:] = tmp
	# Save pickle file
	pickle.dump(thumbs, open(sys.argv[4]+'.pkl', 'wb' ))
else:
	# load pickle file
	thumbs = pickle.load( open(sys.argv[4]+'.pkl', 'rb' ))
	
factor = 1
for f in range(0, scales):
	print('Building mosaic scale', factor)
	result, distMap = computeLayer(img, thumbs, thumbs.shape[0], result, distMap, factor)
	factor = factor * 2

print('Mean:% 3.2f STD:% 3.2f' % (np.mean(distMap), np.std(distMap)))
print('Saving image')
invMap = np.ones((img.shape[0], img.shape[1], 3), float)
distMapRGB = np.zeros((img.shape[0], img.shape[1], 3), float)
for i in range(0, 3):
	distMapRGB[:,:,i] = distMap[:,:]/255.0
cv2.imwrite(sys.argv[3], result*(invMap-distMapRGB)+img*distMapRGB)
print('Done')