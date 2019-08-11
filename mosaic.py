import os, sys, cv2, math, re
import numpy as np

step = 128
tbSize = 16
minError = 48

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
			dist = 256
			index = 0
			for m in range(0, k):
				dist_tmp = np.sum(np.abs(patch-thumbs[m,:,:,:]))/(tbSize * tbSize * 3)
				if dist_tmp < dist:
					dist = dist_tmp
					index = m

			tmp = cv2.imread(sys.argv[2]+filenames[index], cv2.IMREAD_COLOR)
			tmp = getCenterSquare(tmp)
			if step == 1:
				tmp = cv2.blur(tmp,(5,5))
			else:
				tmp = cv2.blur(tmp,(3,3))
			tmp = cv2.resize(tmp, (stepf, stepf))
			if (dist < np.mean(distMap[i:i+stepf,j:j+stepf])):
				distMap[i:i+stepf,j:j+stepf] = dist
				if dist < minError:
					result[i:i+stepf,j:j+stepf,:] = tmp
				else:
					result[i:i+stepf,j:j+stepf,:] = img[i:i+stepf,j:j+stepf,:]

	return result, distMap

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
times = int(round(8196.0/np.max(img.shape[0:2])))
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
		tmp = getCenterSquare(tmp)
		tmp = cv2.blur(tmp,(9,9))
		tmp = cv2.resize(tmp, (tbSize, tbSize))
		thumbs[k,:,:,:] = tmp
		k += 1

	factor = 1
	for f in range(0, 3):
		print('Building mosaic scale', factor)
		result, distMap = computeLayer(img, thumbs, k, result, distMap, factor)
		factor = factor * 2
 
	print(np.mean(distMap))
	print('Saving image')
	cv2.imwrite(sys.argv[3], result*1.0+img*0.0)
	cv2.imwrite('dist.jpg', distMap)
	print('Done')