import numpy
from PIL import Image

# open the image and return 3 matrices, each corresponding to one channel (R, G, B channels)
def openImage(imagePath):
    imOrig = Image.open(imagePath)
    im = numpy.array(imOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue, imOrig]

#compress the matrix of a single channel depending of singular values limit
def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed

# Main Program

print('Image Compression using SVD for a school project')

Red, Green, Blue, originalImage = openImage('test.jpg')

# image width and height:
imageWidth = 512
imageHeight = 512

# number of singular values to use for reconstructing the compressed image
singularValuesLimit = 100

# compressing each color
RedCompressed = compressSingleChannel(Red, singularValuesLimit)
GreenCompressed = compressSingleChannel(Green, singularValuesLimit)
BlueCompressed = compressSingleChannel(Blue, singularValuesLimit)

# creating the new images
imr = Image.fromarray(RedCompressed, mode=None)
img = Image.fromarray(GreenCompressed, mode=None)
imb = Image.fromarray(BlueCompressed, mode=None)

# create the new image
newImage = Image.merge("RGB", (imr, img, imb))

# show both images
originalImage.show()
newImage.show()

# Calculate the compression percentage
mr = imageHeight
mc = imageWidth

originalSize = mr * mc * 3
compressedSize = singularValuesLimit * (1 + mr + mc) * 3

print('original size of the image:')
print(originalSize)

print('compressed size of the image:')
print(compressedSize)

ratio = compressedSize * 1.0 / originalSize

print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
