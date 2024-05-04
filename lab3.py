import cv2
import numpy as np


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


def divide_into_blocks(img, block_size=8):
    """
    Dzieli obraz na bloki o rozmiarze block_size x block_size.
    """
    img_height, img_width = img.shape[:2]
    blocks = []
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            block = img[y:y + block_size, x:x + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append(block)
    return blocks


def zigzag(input):
    """
    Przeprowadza przekształcenie ZigZag dla bloku 8x8.
    """
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    i = 0
    output = np.zeros((vmax * hmax))
    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:  # idąc do góry
            if (v == vmin):
                output[i] = input[v, h]  # na górze
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == hmax - 1) and (v < vmax)):  # z prawej strony
                output[i] = input[v, h]
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax - 1)):  # pośrodku w górę
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # idąc w dół
            if ((v == vmax - 1) and (h <= hmax - 1)):  # na dole
                output[i] = input[v, h]
                h = h + 1
                i = i + 1
            elif (h == hmin):  # z lewej strony
                output[i] = input[v, h]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < vmax - 1) and (h > hmin)):  # pośrodku w dół
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
    if (v == vmax - 1) and (h == hmax - 1):
        output[i] = input[v, h]
    return output

def merge_and_scale_back(y, cr, cb):
    # Skalujemy Cr i Cb do rozmiaru Y
    cr_scaled = cv2.resize(cr, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    cb_scaled = cv2.resize(cb, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Scalanie w pełnokolorowy obraz YCbCr
    return cv2.merge([y, cr_scaled, cb_scaled])

def convert_ycbcr_to_bgr(ycbcr_img):
    return cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)


def task_2_4():
    img = cv2.imread("rgb_color_transition.png")
    ycbcr_img = convert_rgb_to_ycbcr(img)
    y, cr_subsampled, cb_subsampled = subsample_cb_cr(ycbcr_img)

    # Teraz mamy Y, Cr i Cb jako osobne komponenty, Cr i Cb są subsampled
    # Blokowe przetwarzanie Y
    y_blocks = divide_into_blocks(y)

    y_zigzag_blocks = [zigzag(block) for block in y_blocks]

    ycbcr_full = merge_and_scale_back(y, cr_subsampled, cb_subsampled)

    bgr_img = convert_ycbcr_to_bgr(ycbcr_full)

    # Zapisujemy obraz do formatu JPEG
    cv2.imwrite("image.jpeg", bgr_img)

def task_2_5():
    print("ok")

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
