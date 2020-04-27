# python mosaic.py inputImage.jpg imageDatasetPath outputImage.png
import os, sys, cv2, math, re
import numpy as np

scales = 4
step = 64 #Image patch size
tbSize = 32
minError = 64
maxOutSize = 8192
gray = True # process the image in RGB or gray

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

def computeLayer(img, thumbs, k, result, distMap, factor):
	stepf = step * factor
	for i in range(0, img.shape[0]-stepf-1, stepf):
		for j in range(0, img.shape[1]-stepf-1, stepf):
			patch = img[i:i+stepf,j:j+stepf,:]
			patch = cv2.resize(patch, (tbSize, tbSize))
			patch = cv2.medianBlur(patch, 3)
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

	return result, distMap

# img = reference image
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
if gray:
	img = covert2gray(img)
times = int(round(maxOutSize/np.max(img.shape[0:2])))
img = cv2.resize(img, (int(img.shape[1]*times), int(img.shape[0]*times)))

result = np.zeros((img.shape[0], img.shape[1], 3), int)
distMap = np.ones((img.shape[0], img.shape[1]), float)*256

print(sys.argv[1])

# files_list = os.listdir(cfg.data_path)
for (dirpath, dirnames, filenames) in os.walk(sys.argv[2]):
	thumbs = np.zeros((len(filenames), tbSize, tbSize, 3), float)
	k = 0
	print('Rading images')
	for file in filenames:
		tmp = cv2.imread(sys.argv[2]+file , cv2.IMREAD_COLOR)
		if gray:
			tmp = covert2gray(tmp)
		#print(sys.argv[2]+file)
		tmp = getCenterSquare(tmp)
		tmp = cv2.medianBlur(tmp, 15)
		tmp = cv2.resize(tmp, (tbSize, tbSize))
		tmp = cv2.medianBlur(tmp, 3)
		thumbs[k,:,:,:] = tmp
		k += 1

	factor = 1
	for f in range(0, scales):
		print('Building mosaic scale', factor)
		result, distMap = computeLayer(img, thumbs, k, result, distMap, factor)
		factor = factor * 2
 
	print(np.mean(distMap), np.std(distMap))
	print('Saving image')
	cv2.imwrite('dist.jpg', distMap)
	invMap = np.ones((img.shape[0], img.shape[1], 3), float)
	distMapRGB = np.zeros((img.shape[0], img.shape[1], 3), float)
	for i in range(0, 3):
		distMapRGB[:,:,i] = distMap[:,:]/255.0
	cv2.imwrite(sys.argv[3], result*(invMap-distMapRGB)+img*distMapRGB)
	cv2.imwrite('tmp.png', result)
	print('Done')