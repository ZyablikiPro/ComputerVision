import cv2
import numpy

def caption(img, caption):
	imgc = img.copy()

	if type(caption) is list:
		for i in xrange(len(caption)):
			cv2.putText(imgc, str(caption[i]), (10,25 + i * 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
	else:
		cv2.putText(imgc, caption, (10,25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

	return imgc


img_path = 'lenna.png'
scale = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=scale, fy=scale) 


kernels = []
positions = []

kernels.append([[0,0,0],[0,1,0],[0,0,0]])
kernels.append([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernels.append([[0,0,-1],[0,2,0],[-1,0,0]])	#diagonal
kernels.append([[-1,0,0],[0,2,0],[0,0,-1]])
kernels.append([[-1,0,-1],[0,4,0],[-1,0,-1]])	#both diagonal
kernels.append([[0,0,0],[-1,2,-1],[0,0,0]])	#horisontal
kernels.append([[0,-1,0],[0,2,0],[0,-1,0]])	#vertical
kernels.append([[0,-1,0],[-1,4,-1],[0,-1,0]])	#vertical + horisontal

results = []
for kernel in kernels:
	kernel_m = numpy.asarray(kernel, numpy.float32)
	tmp = cv2.filter2D(img, -1, kernel_m)
	tmp = caption(tmp, kernel)
	results.append(tmp)

result = numpy.vstack(( numpy.hstack((results[0:4])), numpy.hstack((results[4:8])) ))

cv2.imwrite('rolls.png', result)
cv2.imshow('roll', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
