import cv2
import numpy as np
from scipy.fftpack import dct
import os
from PIL import Image

def convert_rgb_to_ycbcr(img):
    """
    Konwertuje obraz z RGB do YCbCr.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def subsample_cb_cr(img, sample_type='4:2:0'):
    """
    Próbkowanie składowych Cb i Cr.
    """
    y, cr, cb = cv2.split(img)
    cr_subsampled = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2))
    cb_subsampled = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2))
    return y, cr_subsampled, cb_subsampled

def block_splitting(img):
    """
    Podział obrazu na bloki o rozmiarze 8x8.
    """
    blocks = []
    height, width = img.shape[:2]
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = img[y:y+8, x:x+8]
            blocks.append(block)
    return blocks

def zigzag_encode(block):
    """
    Zwinięcie bloku 8x8 do wiersza 1x64 – algorytm ZigZag.
    """
    return np.array([block[i, j] for (i, j) in zigzag_indices(8)])

def zigzag_indices(n):
    """
    Generuje indeksy dla algorytmu ZigZag.
    """
    indices = [(0, 0)]
    i, j = 0, 0
    up = True
    while len(indices) < n * n:
        if up:
            if i - 1 >= 0 and j + 1 < n:
                i -= 1
                j += 1
            else:
                if j + 1 < n:
                    j += 1
                else:
                    i += 1
                up = False
        else:
            if i + 1 < n and j - 1 >= 0:
                i += 1
                j -= 1
            else:
                if i + 1 < n:
                    i += 1
                else:
                    j += 1
                up = True
        indices.append((i, j))
    return indices

def pad_to_multiple(array, target_shape):
    """
    Dopełnienie tablicy do wielokrotności podanej postaci docelowej.
    """
    pad_width = [(0, 0) if a % b == 0 else (0, b - a % b) for a, b in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant')

def jpeg_compress(blocks):
    """
    Kompresja bloków przy użyciu algorytmu DCT i kwantyzacji.
    """
    quantization_matrix_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                      [12, 12, 14, 19, 26, 58, 60, 55],
                                      [14, 13, 16, 24, 40, 57, 69, 56],
                                      [14, 17, 22, 29, 51, 87, 80, 62],
                                      [18, 22, 37, 56, 68, 109, 103, 77],
                                      [24, 35, 55, 64, 81, 104, 113, 92],
                                      [49, 64, 78, 87, 103, 121, 120, 101],
                                      [72, 92, 95, 98, 112, 100, 103, 99]])

    quantization_matrix_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                         [18, 21, 26, 66, 99, 99, 99, 99],
                                         [24, 26, 56, 99, 99, 99, 99, 99],
                                         [47, 66, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99]])

    compressed_blocks = []
    for block in blocks:
        # Sprawdzenie kształtu bloku
        if block.shape != (8, 8):
            # Jeśli kształt nie jest 8x8, należy go przekształcić
            padded_block = pad_to_multiple(block, (8, 8))
            # Dyskretna transformacja kosinusowa (DCT)
            dct_block = dct(dct(padded_block.T, norm='ortho').T, norm='ortho')
            # Wybór odpowiedniej macierzy kwantyzacji
            quantization_matrix = quantization_matrix_y if block.shape[0] == 8 else quantization_matrix_cbcr
        else:
            # Wybór odpowiedniej macierzy kwantyzacji
            quantization_matrix = quantization_matrix_y
            # Dyskretna transformacja kosinusowa (DCT)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        # Powielenie macierzy kwantyzacji
        quantization_matrix_replicated = np.tile(quantization_matrix, (8, 8))
        # Kwotowanie
        quantized_block = np.round(dct_block / quantization_matrix_replicated)
        compressed_blocks.append(quantized_block)
    return compressed_blocks


def jpeg_compress_full(blocks):
    """
    Kompresja bloków przy użyciu algorytmu DCT i kwantyzacji.
    """
    quantization_matrix_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                      [12, 12, 14, 19, 26, 58, 60, 55],
                                      [14, 13, 16, 24, 40, 57, 69, 56],
                                      [14, 17, 22, 29, 51, 87, 80, 62],
                                      [18, 22, 37, 56, 68, 109, 103, 77],
                                      [24, 35, 55, 64, 81, 104, 113, 92],
                                      [49, 64, 78, 87, 103, 121, 120, 101],
                                      [72, 92, 95, 98, 112, 100, 103, 99]])

    quantization_matrix_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                         [18, 21, 26, 66, 99, 99, 99, 99],
                                         [24, 26, 56, 99, 99, 99, 99, 99],
                                         [47, 66, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99],
                                         [99, 99, 99, 99, 99, 99, 99, 99]])

    compressed_blocks = []
    for block in blocks:
        # Sprawdzenie kształtu bloku
        if block.shape != (8, 8):
            # Jeśli kształt nie jest 8x8, należy go przekształcić
            padded_block = pad_to_multiple(block, (8, 8))
            # Dyskretna transformacja kosinusowa (DCT)
            dct_block = dct(dct(padded_block.T, norm='ortho').T, norm='ortho')
            # Wybór odpowiedniej macierzy kwantyzacji
            quantization_matrix = quantization_matrix_y if block.shape[0] == 8 else quantization_matrix_cbcr
        else:
            # Wybór odpowiedniej macierzy kwantyzacji
            quantization_matrix = quantization_matrix_y
            # Dyskretna transformacja kosinusowa (DCT)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # 4. Wykonanie dyskretnej transformacji cosinusowej na każdym bloku obrazu.
        # Zaimplementowane powyżej.

        # 5. Podzielenie każdego bloku obrazu przez macierz kwantyzacji.
        quantized_block = np.round(dct_block / quantization_matrix)

        # 6. Zaokrąglenie wartości w każdym bloku do liczb całkowitych.
        quantized_block = quantized_block.astype(np.int16)

        compressed_blocks.append(quantized_block)
    return compressed_blocks


def get_file_size(blocks):
    """
    Oblicza rozmiar (liczbę bajtów) bloków JPEG.
    """
    total_size = 0
    for block in blocks:
        block_flattened = block.flatten().astype(np.int16)
        diffs = np.diff(block_flattened)
        total_size += len(diffs) * 2  # Rozmiar każdego elementu to 2 bajty
    return total_size


def save_jpeg(blocks, filename, quality=50):
    """
    Zapisanie bloków do pliku JPEG.
    """
    with open(filename, 'wb') as f:
        f.write(b'\xFF\xD8')  # Nagłówek JPEG
        # Zakodowanie bloków
        for block in blocks:
            block_flattened = block.flatten().astype(np.int16)
            # Algorytm prosty - kodowanie różnicowe dla wartości bez znaku
            diffs = np.diff(block_flattened)
            bytes_ = bytearray()
            for diff in diffs:
                if isinstance(diff, int):
                    diff = abs(diff)
                    bytes_ += diff.to_bytes(2, byteorder='big', signed=False)
                else:
                    diff = abs(diff.item())
                    bytes_ += diff.to_bytes(2, byteorder='big', signed=False)
            f.write(bytes_)
        f.write(b'\xFF\xD9')  # Koniec pliku JPEG


def task_2_4():
    img = cv2.imread("rgb_color_transition.png")
    # Wyświetlanie rozmiaru obrazu wejściowego PNG
    png_size = os.path.getsize("rgb_color_transition.png")
    print("Rozmiar obrazu PNG:", png_size, "bajtów")

    # 1. Konwersja modelu barw: RGB -> YCbCr
    # 2. Przeskalowanie w dół (stratne) macierzy składowych Cb i Cr.
    ycbcr_img = convert_rgb_to_ycbcr(img)
    y, cr_subsampled, cb_subsampled = subsample_cb_cr(ycbcr_img)

    # 3. Podział obrazu na bloki o rozmiarze 8x8.
    cr_blocks = block_splitting(pad_to_multiple(cr_subsampled, (8, 8)))
    cb_blocks = block_splitting(pad_to_multiple(cb_subsampled, (8, 8)))
    y_blocks = block_splitting(pad_to_multiple(y, (8, 8)))

    # 7. Zwinięcie każdego bloku 8x8 do wiersza 1x64 – algorytm ZigZag.
    cr_zigzag = [zigzag_encode(block) for block in cr_blocks]
    cb_zigzag = [zigzag_encode(block) for block in cb_blocks]
    y_zigzag = [zigzag_encode(block) for block in y_blocks]

    # 8. Zakodowanie danych obrazu – m.in. algorytmem Huffmana.
    compressed_cr = jpeg_compress(cr_zigzag)
    compressed_cb = jpeg_compress(cb_zigzag)
    compressed_y = jpeg_compress(y_zigzag)

    # Zapisanie do pliku JPEG
    save_jpeg(compressed_cr + compressed_cb + compressed_y, "compressed_image.jpeg")

    # Pomiar rozmiaru obrazu JPEG
    jpeg_size = get_file_size(compressed_cr + compressed_cb + compressed_y)
    print("Rozmiar obrazu JPEG:", jpeg_size, "bajtów")


def task_2_5():
    img = cv2.imread("rgb_color_transition.png")
    # 1. Konwersja modelu barw: RGB -> YCbCr
    ycbcr_img = convert_rgb_to_ycbcr(img)
    # 2. Przeskalowanie w dół (stratne) macierzy składowych Cb i Cr.
    y, cr_subsampled, cb_subsampled = subsample_cb_cr(ycbcr_img)

    # 3. Podział obrazu na bloki o rozmiarze 8x8.
    cr_blocks = block_splitting(pad_to_multiple(cr_subsampled, (8, 8)))
    cb_blocks = block_splitting(pad_to_multiple(cb_subsampled, (8, 8)))
    y_blocks = block_splitting(pad_to_multiple(y, (8, 8)))

    # 4. Wykonanie dyskretnej transformacji cosinusowej na każdym bloku obrazu
    cr_dct = [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in cr_blocks]
    cb_dct = [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in cb_blocks]
    y_dct = [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in y_blocks]

    # 5. Podzielenie każdego bloku obrazu przez macierz kwantyzacji.
    cr_quantized = jpeg_compress_full(cr_dct)
    cb_quantized = jpeg_compress_full(cb_dct)
    y_quantized = jpeg_compress_full(y_dct)

    # 6. Zaokrąglenie wartości w każdym bloku do liczb całkowitych.
    cr_quantized = [block.astype(np.int16) for block in cr_quantized]
    cb_quantized = [block.astype(np.int16) for block in cb_quantized]
    y_quantized = [block.astype(np.int16) for block in y_quantized]

    # 7. Zwinięcie każdego bloku 8x8 do wiersza 1x64 – algorytm ZigZag.
    cr_zigzag = [zigzag_encode(block) for block in cr_quantized]
    cb_zigzag = [zigzag_encode(block) for block in cb_quantized]
    y_zigzag = [zigzag_encode(block) for block in y_quantized]

    # 8. Zakodowanie danych obrazu – m.in. algorytmem Huffmana.
    compressed_cr = jpeg_compress(cr_zigzag)
    compressed_cb = jpeg_compress(cb_zigzag)
    compressed_y = jpeg_compress(y_zigzag)

    # Zapisanie do pliku JPEG
    save_jpeg(compressed_cr + compressed_cb + compressed_y, "compressed_image_full.jpeg")



while True:
    print("\nMenu:")
    print("1. Zadanie 4 Lista 2")
    print("2. Zadanie 5 Lista 2")
    print("0. Wyjście")

    choice = input("Wybierz numer zadania (lub 0 aby wyjść): ")

    if choice == '1':
        task_2_4()
    elif choice == '2':
        task_2_5()
    elif choice == '0':
        print("Koniec programu.")
        break
    else:
        print("Niepoprawny wybór. Wybierz numer od 1 do 5 lub 0 aby wyjść.")
