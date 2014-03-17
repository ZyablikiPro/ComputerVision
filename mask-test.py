import cv2
import numpy

def drow_image(img, dst, row, col, caption):
	cv2.putText(img, caption, (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
	dst[img.shape[0] * row:img.shape[0] * (row + 1), img.shape[1] * col:img.shape[1] * (col + 1)] = img

def create_image(img, sx, sy):
	dst_size = img.shape[0] * sx, img.shape[1] * sy, img.shape[2]
	dst = numpy.zeros(dst_size, img.dtype)
	return dst

kernel_sizes = [1, 3, 9, 27]
img_path = 'lenna.png'
scale = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
result = create_image(img, 3, len(kernel_sizes))

for i in xrange(len(kernel_sizes)):
	kernel = numpy.ones((kernel_sizes[i], kernel_sizes[i]), numpy.float32)/kernel_sizes[i]**2
	tmp = cv2.filter2D(img, -1, kernel)
	drow_image(tmp, result, 0, i, "averaging, {a}x{a}".format(a = kernel_sizes[i]))

	kernel = cv2.getGaussianKernel(kernel_sizes[i], -1)
	tmp = cv2.filter2D(img, -1, kernel)
	drow_image(tmp, result, 1, i, "gaussian, {a}x{a}".format(a = kernel_sizes[i]))

	tmp = cv2.medianBlur(img, kernel_sizes[i])
	drow_image(tmp, result, 2, i, "median, {a}x{a}".format(a = kernel_sizes[i]))


cv2.imshow('w1', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


#img = cv2.Sobel(img, -1, 1, 0)

#kernel = [[-1,0,-1],[0,4,0],[-1,0,-1]]
#kernel = numpy.asarray(kernel, numpy.float32)
#tmp = cv2.filter2D(img, -1, kernel)
#cv2.imshow('w2', tmp)



#mask = cv2.getGaussianKernel(5, -1)

#img = cv2.filter2D(img, -1, mask)
#cv2.imshow('gaussian', cv2.filter2D(img, -1, mask))
