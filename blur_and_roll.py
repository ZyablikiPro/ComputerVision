import cv2
import numpy

color = (0, 255, 0)

def caption(img, caption):
	imgc = img.copy()

	if type(caption) is list:
		for i in xrange(len(caption)):
			cv2.putText(imgc, str(caption[i]), (10,25 + i * 25), cv2.FONT_HERSHEY_PLAIN, 1, color)
	else:
		cv2.putText(imgc, caption, (10,25), cv2.FONT_HERSHEY_PLAIN, 1, color)

	return imgc


blur_kernel_sizes = [1, 3, 9, 27]
img_path = 'lenna_gauss.jpg'
scale = 0.5

img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=scale, fy=scale) 

average = []
gaussian = []
median = []

for i in xrange(len(blur_kernel_sizes)):
	roll_kernel = [[0,-1,0],[-1,4,-1],[0,-1,0]]	#vertical + horisontal
	roll_kernel_m = numpy.asarray(roll_kernel, numpy.float32)

	kernel = numpy.ones((blur_kernel_sizes[i], blur_kernel_sizes[i]), numpy.float32)/blur_kernel_sizes[i]**2
	tmp = cv2.filter2D(img, -1, kernel)
	tmp = cv2.filter2D(tmp, -1, roll_kernel_m)
	tmp = caption(tmp, "average, {a}x{a}".format(a = blur_kernel_sizes[i]))
	average.append(tmp)

	kernel = cv2.getGaussianKernel(blur_kernel_sizes[i], -1)
	tmp = cv2.filter2D(img, -1, kernel)
	tmp = cv2.filter2D(tmp, -1, roll_kernel_m)
	tmp = caption(tmp, "gaussian, {a}x{a}".format(a = blur_kernel_sizes[i]))
	gaussian.append(tmp)

	tmp = cv2.medianBlur(img, blur_kernel_sizes[i])
	tmp = cv2.filter2D(tmp, -1, roll_kernel_m)
	tmp = caption(tmp, "median, {a}x{a}".format(a = blur_kernel_sizes[i]))
	median.append(tmp)

result = numpy.vstack(( numpy.hstack((average[:])), numpy.hstack((gaussian[:])), numpy.hstack((median[:])) ))

cv2.imshow('w1', result)
cv2.imwrite('blur_roll_gauss.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
