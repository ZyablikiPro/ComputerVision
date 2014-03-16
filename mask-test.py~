import cv2
import numpy

core_sizes = [1, 3, 9, 27]
img_path = 'lenna.png'
scale = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
result_size = img.shape[0], img.shape[1] * len(core_sizes), img.shape[2]
result = numpy.zeros(result_size, img.dtype)

for i in xrange(len(core_sizes)):
	mask = cv2.getGaussianKernel(core_sizes[i], -1)
	tmp = cv2.filter2D(img, -1, mask)
	cv2.putText(tmp,"gaussian, {a}x{a}".format(a = core_sizes[i]), (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
	result[0:img.shape[0], img.shape[1] * i:img.shape[1] * (i + 1)] = tmp


cv2.imshow('w1', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


#img = cv2.Sobel(img, -1, 1, 0)

#mask = [[-1,0,1],[-2,0,2],[-1,0,1]]
#mask = numpy.asarray(mask, numpy.float32)
#mask = cv2.getGaussianKernel(5, -1)

#img = cv2.filter2D(img, -1, mask)
#cv2.imshow('gaussian', cv2.filter2D(img, -1, mask))
