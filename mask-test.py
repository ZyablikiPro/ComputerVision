import cv2
import numpy

def drow_image(img, dst, position, caption):
	imgc = img.copy()

	if type(caption) is list:
		for i in xrange(len(caption)):
			cv2.putText(imgc, str(caption[i]), (10,25 + i * 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
	else:
		cv2.putText(imgc, caption, (10,25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
	
	dst[imgc.shape[0] * position[0]:imgc.shape[0] * (position[0] + 1), imgc.shape[1] * position[1]:imgc.shape[1] * (position[1] + 1)] = imgc

def create_image(img, sx, sy):
	dst_size = img.shape[0] * sx, img.shape[1] * sy, img.shape[2]
	dst = numpy.zeros(dst_size, img.dtype)
	return dst

blur_kernel_sizes = [1, 3, 9, 27]
img_path = 'lenna.png'
scale = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
result = create_image(img, 3, len(blur_kernel_sizes))

for i in xrange(len(blur_kernel_sizes)):
	kernel = numpy.ones((blur_kernel_sizes[i], blur_kernel_sizes[i]), numpy.float32)/blur_kernel_sizes[i]**2
	tmp = cv2.filter2D(img, -1, kernel)
	drow_image(tmp, result, [0, i], "averaging, {a}x{a}".format(a = blur_kernel_sizes[i]))

	kernel = cv2.getGaussianKernel(blur_kernel_sizes[i], -1)
	tmp = cv2.filter2D(img, -1, kernel)
	drow_image(tmp, result, [1, i], "gaussian, {a}x{a}".format(a = blur_kernel_sizes[i]))

	tmp = cv2.medianBlur(img, blur_kernel_sizes[i])
	drow_image(tmp, result, [2, i], "median, {a}x{a}".format(a = blur_kernel_sizes[i]))

cv2.imshow('w1', result)


result2 = create_image(img, 2, 4)

kernels = []
positions = []

kernels.append([[0,0,0],[0,1,0],[0,0,0]])
positions.append([0, 0])

kernels.append([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
positions.append([1, 0])

kernels.append([[0,0,-1],[0,2,0],[-1,0,0]])	#diagonal
positions.append([0, 1])

kernels.append([[-1,0,0],[0,2,0],[0,0,-1]])
positions.append([0, 2])

kernels.append([[-1,0,-1],[0,4,0],[-1,0,-1]])	#both diagonal
positions.append([0, 3])

kernels.append([[0,0,0],[-1,2,-1],[0,0,0]])	#horisontal
positions.append([1, 1])

kernels.append([[0,-1,0],[0,2,0],[0,-1,0]])	#vertical
positions.append([1, 2])

kernels.append([[0,-1,0],[-1,4,-1],[0,-1,0]])	#vertical + horisontal
positions.append([1, 3])

for i in xrange(len(kernels)):
	kernel_m = numpy.asarray(kernels[i], numpy.float32)
	tmp = cv2.filter2D(img, -1, kernel_m)
	drow_image(tmp, result2, positions[i], kernels[i])	

cv2.imshow('w2', result2)


cv2.waitKey(0)
cv2.destroyAllWindows()