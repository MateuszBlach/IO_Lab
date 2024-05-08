import cv2
import numpy
from scipy.fftpack import dct
from scipy.fftpack import idct
import struct
import zlib

#
# PPM file header
#
ppm_ascii_header = 'P3\n3 3\n255\n'  # TODO: implement
ppm_binary_header = 'P6\n3 3\n255\n'  # TODO: implement

#
# Image data
#
image = numpy.array([82, 71, 66, 82, 71, 66, 82, 71, 66, 82, 71, 66, 82, 71, 66, 82, 71, 66,
    82, 71, 66, 82, 71, 66, 82, 71, 66,], dtype=numpy.uint8)  # TODO: implement

#
# Save the PPM image as an ASCII file
#
with open('lab4-ascii.ppm', 'w') as fh:
    fh.write(ppm_ascii_header)
    image.tofile(fh, sep=' ')
    fh.write('\n')

#
# Save the PPM image as a binary file
#
with open('lab4-binary.ppm', 'wb') as fh:
    fh.write(bytearray(ppm_binary_header, 'ascii'))
    image.tofile(fh)


#
ppm_header = 'P3\n120 8\n255\n'
image = numpy.array([0, 0, 0], dtype=numpy.uint8)
dummy = numpy.array([0, 0, 0], dtype=numpy.uint8)

step = 15  # 255/17

for i in range(0, 17):
    dummy[2] += step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[1] += step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[2] -= step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[0] += step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[1] -= step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[2] += step
    image = numpy.append(image, dummy)

for i in range(0, 17):
    dummy[1] += step
    image = numpy.append(image, dummy)

line = numpy.copy(image)
image = numpy.append(image, image)
image = numpy.append(image, image)
image = numpy.append(image, image)

with open('lab4-rainbow.ppm', 'w') as fh:
    fh.write(ppm_header)
    image.tofile(fh, sep=' ')
    fh.write('\n')
#


#
# Image data
image = numpy.array([0], dtype=numpy.uint8)
image = numpy.append(image, line)
image = image.reshape(-1, 1)
image = numpy.append(image, image)
image = numpy.append(image, image)
image = numpy.append(image, image)

#
# Construct signature
#
png_file_signature = b'\x89PNG\r\n\x1a\n'  # TODO: implement

#
# Construct header
#
header_id = b'IHDR'  # TODO: implement
header_content = b'\x00\x00\x00\x78\x00\x00\x00\x08\x08\x02\x00\x00\x00'  # TODO: implement
header_size = struct.pack('!I', len(header_content))  # TODO: implement
header_crc = struct.pack('!I', zlib.crc32(header_id + header_content))  # TODO: implement
png_file_header = header_size + header_id + header_content + header_crc

#
# Construct data
#
data_id = b'IDAT'  # TODO: implement
data_content = zlib.compress(image,0)  # TODO: implement
data_size = struct.pack('!I', len(data_content))  # TODO: implement
data_crc = struct.pack('!I', zlib.crc32(data_id + data_content))  # TODO: implement
png_file_data = data_size + data_id + data_content + data_crc

#
# Consruct end
#
end_id = b'IEND'
end_content = b''
end_size = struct.pack('!I', len(end_content))
end_crc = struct.pack('!I', zlib.crc32(end_id + end_content))
png_file_end = end_size + end_id + end_content + end_crc

#
# Save the PNG image as a binary file
#
with open('lab4.png', 'wb') as fh:
    fh.write(png_file_signature)
    fh.write(png_file_header)
    fh.write(png_file_data)
    fh.write(png_file_end)



#
# 2d Discrete Cosinus Transform
#
def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


#
# Calculate quantisation matrices
#
# Based on: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
#           #step-3-and-4-discrete-cosinus-transform-and-quantisation
#
_QY = numpy.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 48, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

_QC = numpy.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])


def _scale(QF):
    if QF < 50 and QF >= 1:
        scale = numpy.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError('Quality Factor must be in the range [1..99]')

    scale = scale / 100.0
    return scale


def QY(QF=85):
    return _QY * _scale(QF)


def QC(QF=85):
    return _QC * _scale(QF)


# 0. Zdefiniowanie obrazu
image = numpy.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]], dtype=numpy.uint8)
end = numpy.array([[255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255]], dtype=numpy.uint8)
image = numpy.append(image, line)
image = numpy.append(image, end)
image = numpy.append(image, image)
image = numpy.append(image, image)
image = numpy.append(image, image)
image = image.reshape(8, 128, 3)

# 1. Konwersja RGB do YCbCr

converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# 2. Przeskalowanie w dol Cr i Cb

samp0 = numpy.copy(converted)
samp2Cr = []
samp2Cb = []
samp4Cr = []
samp4Cb = []
for i in range(0, len(converted)):
    for j in range(0, len(converted[0])):
        if (2 * i + j) % 4 == 0:  # [x 0] [0 0]
            samp4Cr.append(converted[i][j][1])  # [0 0] [x 0]
            samp4Cb.append(converted[i][j][2])
        if (i + j) % 2 == 0:
            samp2Cr.append(converted[i][j][1])  # [x 0] [x 0]
            samp2Cb.append(converted[i][j][2])  # [0 x] [0 x]


# 3. Podzial na bloki o rozmiarze 8x8
horizontal = int(len(samp0[0]) / 8)
blocks = int(len(samp0) * horizontal / 8)
blocksY = numpy.zeros([blocks, 8, 8], dtype=numpy.uint8)
blocks0Cr = numpy.zeros([blocks, 8, 8], dtype=numpy.uint8)
blocks0Cb = numpy.zeros([blocks, 8, 8], dtype=numpy.uint8)
blocks2Cr = numpy.zeros([blocks // 2, 8, 8], dtype=numpy.uint8)
blocks2Cb = numpy.zeros([blocks // 2, 8, 8], dtype=numpy.uint8)
blocks4Cr = numpy.zeros([blocks // 4, 8, 8], dtype=numpy.uint8)
blocks4Cb = numpy.zeros([blocks // 4, 8, 8], dtype=numpy.uint8)

x = 0
y = 0
z = 0
for i in range(0, len(samp0)):
    for j in range(0, len(samp0[0])):
        blocksY[x][y][z] = samp0[i][j][0]
        blocks0Cr[x][y][z] = samp0[i][j][1]
        blocks0Cb[x][y][z] = samp0[i][j][2]
        z = (z + 1) % 8
        if z == 0:
            x = (x + 1) % horizontal
    y = (y + 1) % 8

x = 0
y = 0
z = 0
horizontal //= 2
for i in range(0, len(samp2Cr)):
    blocks2Cr[x][y][z] = samp2Cr[i]
    blocks2Cb[x][y][z] = samp2Cb[i]
    z = (z + 1) % 8
    if z == 0:
        x = (x + 1) % horizontal
        if x == 0:
            y = (y + 1) % 8

x = 0
y = 0
z = 0
horizontal //= 2
for i in range(0, len(samp4Cr)):
    blocks4Cr[x][y][z] = samp4Cr[i]
    blocks4Cb[x][y][z] = samp4Cb[i]
    z = (z + 1) % 8
    if z == 0:
        x = (x + 1) % horizontal
        if x == 0:
            y = (y + 1) % 8

# 4. Wykonanie dyskretnej transformacji cosinusowej na kazdym bloku obrazu.
DCTY = []
DCT0Cr = []
DCT0Cb = []
DCT2Cr = []
DCT2Cb = []
DCT4Cr = []
DCT4Cb = []

for i in range(0, len(blocksY)):
    DCTY = numpy.append(DCTY, dct2(numpy.asarray(blocksY[i])))
    DCT0Cr = numpy.append(DCT0Cr, dct2(numpy.asarray(blocks0Cr[i])))
    DCT0Cb = numpy.append(DCT0Cb, dct2(numpy.asarray(blocks0Cb[i])))
DCTY = DCTY.reshape(-1, 8, 8)
DCT0Cr = DCT0Cr.reshape(-1, 8, 8)
DCT0Cb = DCT0Cb.reshape(-1, 8, 8)

for i in range(0, len(blocks2Cr)):
    DCT2Cr = numpy.append(DCT2Cr, dct2(numpy.asarray(blocks2Cr[i])))
    DCT2Cb = numpy.append(DCT2Cb, dct2(numpy.asarray(blocks2Cb[i])))
DCT2Cr = DCT2Cr.reshape(-1, 8, 8)
DCT2Cb = DCT2Cb.reshape(-1, 8, 8)

for i in range(0, len(blocks4Cr)):
    DCT4Cr = numpy.append(DCT4Cr, dct2(numpy.asarray(blocks4Cr[i])))
    DCT4Cb = numpy.append(DCT4Cb, dct2(numpy.asarray(blocks4Cb[i])))
DCT4Cr = DCT4Cr.reshape(-1, 8, 8)
DCT4Cb = DCT4Cb.reshape(-1, 8, 8)

# 5. Podzielenie kazdego bloku obrazu przez macierz kwantyzacji.
div0 = QY()
div1 = QC()

divY = []
div0Cr = []
div0Cb = []
div2Cr = []
div2Cb = []
div4Cr = []
div4Cb = []

for i in range(0, len(DCTY)):
    divY = numpy.append(divY, numpy.divide(DCTY[i], div0))
    div0Cr = numpy.append(div0Cr, numpy.divide(DCT0Cr[i], div1))
    div0Cb = numpy.append(div0Cb, numpy.divide(DCT0Cb[i], div1))
divY = divY.reshape(-1, 8, 8)
div0Cr = div0Cr.reshape(-1, 8, 8)
div0Cb = div0Cb.reshape(-1, 8, 8)

for i in range(0, len(DCT2Cr)):
    div2Cr = numpy.append(div2Cr, numpy.divide(DCT2Cr[i], div1))
    div2Cb = numpy.append(div2Cb, numpy.divide(DCT2Cb[i], div1))
div2Cr = div2Cr.reshape(-1, 8, 8)
div2Cb = div2Cb.reshape(-1, 8, 8)

for i in range(0, len(DCT4Cb)):
    div4Cr = numpy.append(div4Cr, numpy.divide(DCT4Cr[i], div1))
    div4Cb = numpy.append(div4Cb, numpy.divide(DCT4Cb[i], div1))
div4Cr = div4Cr.reshape(-1, 8, 8)
div4Cb = div4Cb.reshape(-1, 8, 8)

# 6. Zaokraglenie wartosci w kazdym bloku do liczb calkowitych.

blockY = numpy.asarray(divY).round()
block0Cr = numpy.asarray(div0Cr).round()
block0Cb = numpy.asarray(div0Cb).round()
block2Cr = numpy.asarray(div2Cr).round()
block2Cb = numpy.asarray(div2Cb).round()
block4Cr = numpy.asarray(div4Cr).round()
block4Cb = numpy.asarray(div4Cb).round()


# 7. Zwiniecie kazdego bloku 8x8 do wiersza 1x64 – algorytm ZigZag
def zigZag(source):
    temp = numpy.zeros(64, dtype=numpy.uint8)
    n = 0
    x = 0
    y = 0
    while n < 63:
        while y < 8 and x > -1:
            temp[n] = source[x][y]
            y += 1  # -\
            x -= 1  # /
            n += 1  # /
        x += 1
        if y == 8:
            y -= 1
            x += 1
        while x < 8 and y > -1:
            temp[n] = source[x][y]
            y -= 1  # /
            x += 1  # /
            n += 1  # \-
        y += 1
        if x == 8:
            x -= 1
            y += 1
    temp[63] = source[7][7]
    return temp


zigZagY = []
zigZag0Cr = []
zigZag0Cb = []
zigZag2Cr = []
zigZag2Cb = []
zigZag4Cr = []
zigZag4Cb = []
for i in range(0, len(blockY)):
    zigZagY = numpy.append(zigZagY, zigZag(blockY[i]))
    zigZag0Cr = numpy.append(zigZag0Cr, zigZag(block0Cr[i]))
    zigZag0Cb = numpy.append(zigZag0Cb, zigZag(block0Cb[i]))
for i in range(0, len(blocks2Cr)):
    zigZag2Cr = numpy.append(zigZag2Cr, zigZag(block2Cr[i]))
    zigZag2Cb = numpy.append(zigZag2Cb, zigZag(block2Cb[i]))
for i in range(0, len(blocks4Cb)):
    zigZag4Cr = numpy.append(zigZag4Cr, zigZag(block4Cr[i]))
    zigZag4Cb = numpy.append(zigZag4Cb, zigZag(block4Cb[i]))


# 8. Zakodowanie danych obrazu
flatY = zigZagY.flatten()
flat0Cr = zigZag0Cr.flatten()
flat0Cb = zigZag0Cb.flatten()
flat2Cr = zigZag2Cr.flatten()
flat2Cb = zigZag2Cb.flatten()
flat4Cr = zigZag4Cr.flatten()
flat4Cb = zigZag4Cb.flatten()

im0 = numpy.concatenate((flatY, flat0Cr, flat0Cb))
im2 = numpy.concatenate((flatY, flat2Cr, flat2Cr))
im4 = numpy.concatenate((flatY, flat4Cr, flat4Cb))

comp0 = zlib.compress(im0, 0)
comp2 = zlib.compress(im2, 0)
comp4 = zlib.compress(im4, 0)

print("Rozmiar bez próbkowania: " + str(len(comp0)))
print("Rozmiar z próbkowaniem co drugi element: " + str(len(comp2)))
print("Rozmiar z próbkowaniem co czwarty element: " + str(len(comp4)))

# 7'.
# Pomijamy bo wczesniej Zig Zag zostal zrobiony tylko dla analizy w kroku 8

# 6'.
# Nic nie trzeba robic, wartosci sa juz zaokraglone do int


# 5'.
mulY = []
mul0Cr = []
mul0Cb = []
mul2Cr = []
mul2Cb = []
mul4Cr = []
mul4Cb = []

for i in range(0, len(blockY)):
    mulY = numpy.append(mulY, numpy.multiply(blockY[i], div0))
    mul0Cr = numpy.append(mul0Cr, numpy.multiply(block0Cr[i], div1))
    mul0Cb = numpy.append(mul0Cb, numpy.multiply(block0Cb[i], div1))
mulY = mulY.reshape(-1, 8, 8)
mul0Cr = mul0Cr.reshape(-1, 8, 8)
mul0Cb = mul0Cb.reshape(-1, 8, 8)

for i in range(0, len(block2Cr)):
    mul2Cr = numpy.append(mul2Cr, numpy.multiply(block2Cr[i], div1))
    mul2Cb = numpy.append(mul2Cb, numpy.multiply(block2Cb[i], div1))
mul2Cr = mul2Cr.reshape(-1, 8, 8)
mul2Cb = mul2Cb.reshape(-1, 8, 8)

for i in range(0, len(block4Cb)):
    mul4Cr = numpy.append(mul4Cr, numpy.multiply(block4Cr[i], div1))
    mul4Cb = numpy.append(mul4Cb, numpy.multiply(block4Cb[i], div1))
mul4Cr = mul4Cr.reshape(-1, 8, 8)
mul4Cb = mul4Cb.reshape(-1, 8, 8)

# 4'. Reverse DCT
reverseY = []
reverse0Cr = []
reverse0Cb = []
reverse2Cr = []
reverse2Cb = []
reverse4Cr = []
reverse4Cb = []

for i in range(0, len(blockY)):
    reverseY = numpy.append(reverseY, idct2(mulY[i]))
    reverse0Cr = numpy.append(reverse0Cr, idct2(mul0Cr[i]))
    reverse0Cb = numpy.append(reverse0Cb, idct2(mul0Cb[i]))
reverseY = reverseY.reshape(-1, 8, 8)
reverse0Cr = reverse0Cr.reshape(-1, 8, 8)
reverse0Cb = reverse0Cb.reshape(-1, 8, 8)

for i in range(0, len(block2Cr)):
    reverse2Cr = numpy.append(reverse2Cr, idct2(mul2Cr[i]))
    reverse2Cb = numpy.append(reverse2Cb, idct2(mul2Cb[i]))
reverse2Cr = reverse2Cr.reshape(-1, 8, 8)
reverse2Cb = reverse2Cb.reshape(-1, 8, 8)

for i in range(0, len(block4Cb)):
    reverse4Cr = numpy.append(reverse4Cr, idct2(mul4Cr[i]))
    reverse4Cb = numpy.append(reverse4Cb, idct2(mul4Cb[i]))
reverse4Cr = reverse4Cr.reshape(-1, 8, 8)
reverse4Cb = reverse4Cb.reshape(-1, 8, 8)

# 3'. Combine 8x8 blocks to original image
restoredY = []
restored0Cr = []
restored0Cb = []
restored2Cr = []
restored2Cb = []
restored4Cr = []
restored4Cb = []

i = 0
j = 0
y = 0
while (i + j) < len(reverse4Cr):
    for x in range(0, 8):
        restored4Cr = numpy.append(restored4Cr, reverse4Cr[i + j][y][x])
        restored4Cb = numpy.append(restored4Cb, reverse4Cb[i + j][y][x])
    i = (i + 1) % horizontal
    if i == 0:
        y += 1
        if y == 8:
            y = 0
            j += horizontal
horizontal *= 2

i = 0
j = 0
y = 0
while (i + j) < len(reverse2Cb):
    for x in range(0, 8):
        restored2Cr = numpy.append(restored2Cr, reverse2Cr[i + j][y][x])
        restored2Cb = numpy.append(restored2Cb, reverse2Cb[i + j][y][x])
    i = (i + 1) % horizontal
    if i == 0:
        y += 1
        if y == 8:
            y = 0
            j += horizontal
horizontal *= 2

i = 0
j = 0
y = 0
while (i + j) < len(reverseY):
    for x in range(0, 8):
        restoredY = numpy.append(restoredY, reverseY[i + j][y][x])
        restored0Cr = numpy.append(restored0Cr, reverse0Cr[i + j][y][x])
        restored0Cb = numpy.append(restored0Cb, reverse0Cb[i + j][y][x])
    i = (i + 1) % horizontal
    if i == 0:
        y += 1
        if y == 8:
            y = 0
            j += horizontal


# 2'.
def imagine(Y, Cr, Cb, samp):
    channelled = []
    x = 0
    y = 0
    for n in range(0, len(Y)):
        temp = numpy.zeros(3, dtype=numpy.uint8)
        temp[0] = Y[n]
        temp[1] = Cr[x]
        temp[2] = Cb[x]
        channelled = numpy.append(channelled, temp)
        if samp > 0:
            y = (y + 1) % samp
        if y == 0:
            x += 1
    channelled = channelled.reshape(8, 128, 3)
    channelled = channelled.astype(dtype=numpy.uint8)
    return channelled


image0N = imagine(restoredY, restored0Cr, restored0Cb, 0)
image2N = imagine(restoredY, restored2Cr, restored2Cb, 2)
image4N = imagine(restoredY, restored4Cr, restored4Cb, 4)

# 1'. Convert YCbCr to RGB
image0 = cv2.cvtColor(image0N, cv2.COLOR_YCrCb2RGB)
image2 = cv2.cvtColor(image2N, cv2.COLOR_YCrCb2RGB)
image4 = cv2.cvtColor(image4N, cv2.COLOR_YCrCb2RGB)


# 0'. Zapis obrazu

ppm_head = 'P3\n128 8\n255\n'
with open('lab4-po-kompresji0.ppm', 'w') as fh:
    fh.write(ppm_head)
    image0.tofile(fh, sep=' ')
    fh.write('\n')
with open('lab4-po-kompresji2.ppm', 'w') as fh:
    fh.write(ppm_head)
    image2.tofile(fh, sep=' ')
    fh.write('\n')
with open('lab4-po-kompresji4.ppm', 'w') as fh:
    fh.write(ppm_head)
    image4.tofile(fh, sep=' ')
    fh.write('\n')


image_from0 = cv2.imread('lab4-po-kompresji0.ppm')
image_from2 = cv2.imread('lab4-po-kompresji2.ppm')
image_from4 = cv2.imread('lab4-po-kompresji4.ppm')

def resize(img,factor):
    height, width = img.shape[:2]
    return cv2.resize(img,(width*factor,height*factor))

cv2.imshow('Brak probkowania',resize(image_from0,8))
cv2.imshow('Probkowanie co drugi',resize(image_from2,8))
cv2.imshow('Probkowanie co czwarty',resize(image_from4,8))
cv2.waitKey(0)
cv2.destroyAllWindows()

