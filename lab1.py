import cv2
import numpy as np

def task_1():
    print("Wybrałeś zadanie 1.")
    # Wczytanie obrazu
    image = cv2.imread('widok.jpg')

    # Maska górnoprzepustowa (detektor krawędzi)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # Zastosowanie maski na obrazie za pomocą funkcji filtrującej
    filtered_image = cv2.filter2D(image, -1, kernel)

    # Wyświetlenie oryginalnego i przefiltrowanego obrazu
    cv2.imshow('Oryginalny obraz', image)
    cv2.imshow('Obraz z nalozonym filtrem', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task_2():
    print("Wybrałeś zadanie 2.")
    # Wczytanie obrazu
    image = cv2.imread('widok.jpg')

    # Konwersja kolorów z [0-255;0-255;0-255] na [0-1.0;0-1.0;0-1.0]
    image_float = image.astype(np.float32) / 255.0

    # Macierz przekształcenia kolorów
    transformation_matrix = np.array([[0.393, 0.769, 0.189],
                                      [0.349, 0.689, 0.168],
                                      [0.272, 0.534, 0.131]])

    # Przekształcenie wartości kolorów zgodnie z podanym wzorem
    transformed_image = np.clip(np.dot(image_float, transformation_matrix), 0, 1.0)

    # Wyświetlenie oryginalnego i przekształconego obrazu
    cv2.imshow('Oryginalny obraz', image)
    cv2.imshow('Obraz po transformacji kolorow', (transformed_image * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task_3():
    print("Wybrałeś zadanie 3.")
    # Wczytanie obrazu
    image = cv2.imread('widok.jpg')

    # Macierz przekształcenia kolorów z RGB do YCrCb
    conversion_matrix = np.array([[0.229, 0.587, 0.114],
                                  [0.500, -0.418, -0.082],
                                  [-0.168, -0.331, 0.500]])

    # Stałe dodawane do wyniku
    constants = np.array([0, 128, 128])

    # Przekształcenie wartości kolorów zgodnie z podanym wzorem
    ycbcr_image = np.dot(image, conversion_matrix.T) + constants

    # Ograniczenie wartości kolorów do przedziału [0, 255]
    ycbcr_image = np.clip(ycbcr_image, 0, 255).astype(np.uint8)

    # Wyodrębnienie składowych Y, Cb, Cr
    Y, Cr, Cb = cv2.split(ycbcr_image)

    # Wyświetlenie oryginalnego obrazu i składowych Y, Cb, Cr w odcieniach szarości
    cv2.imshow('Oryginalny obraz', image)
    cv2.imshow('Skladowa Y', Y)
    cv2.imshow('Skladowa Cb', Cb)
    cv2.imshow('Skladowa Cr', Cr)

    # Konwersja z powrotem do przestrzeni kolorowej RGB
    inverse_conversion_matrix = np.linalg.inv(conversion_matrix)
    rgb_image = np.dot(ycbcr_image - constants, inverse_conversion_matrix.T)

    # Ograniczenie wartości kolorów do przedziału [0, 255]
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Wyświetlenie obrazu po konwersji odwrotnej
    cv2.imshow('Obraz po konwersji odwrotnej', rgb_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task_4():
    print("Wybrałeś zadanie 4.")
    # Wczytanie obrazu
    image = cv2.imread('widok.jpg')

    # Macierz przekształcenia kolorów z RGB do YCrCb
    conversion_matrix = np.array([[0.229, 0.587, 0.114],
                                  [0.500, -0.418, -0.082],
                                  [-0.168, -0.331, 0.500]])

    # Stałe dodawane do wyniku
    constants = np.array([0, 128, 128])

    # Przekształcenie wartości kolorów zgodnie z podanym wzorem
    ycbcr_image = np.dot(image, conversion_matrix.T) + constants

    # Ograniczenie wartości kolorów do przedziału [0, 255]
    ycbcr_image = np.clip(ycbcr_image, 0, 255).astype(np.uint8)

    # Wyodrębnienie składowych Y, Cb, Cr
    Y, Cr, Cb = cv2.split(ycbcr_image)

    # Downsampling na kanałach Cb i Cr (usunięcie co drugiego piksela)
    downsampled_Cb = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    downsampled_Cr = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Upsampling na kanałach Cb i Cr (powtórzenie piksela)
    upsampled_Cb = cv2.resize(downsampled_Cb, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
    upsampled_Cr = cv2.resize(downsampled_Cr, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Złożenie obrazu RGB z otrzymanych macierzy Cb, Cr i Y
    reconstructed_ycbcr_image = cv2.merge((Y, upsampled_Cb, upsampled_Cr))

    # Konwersja z powrotem do przestrzeni kolorowej RGB
    inverse_conversion_matrix = np.linalg.inv(conversion_matrix)
    rgb_image = np.dot(reconstructed_ycbcr_image - constants, inverse_conversion_matrix.T)

    # Ograniczenie wartości kolorów do przedziału [0, 255]
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Wyświetlenie oryginalnego obrazu i obrazu po transmisji
    cv2.imshow('Oryginalny obraz', image)
    cv2.imshow('Obraz po transmisji', rgb_image)

    # Wyświetlenie poszczególnych składowych Y, Cb, Cr w odcieniach szarości
    cv2.imshow('Skladowa Y', Y)
    cv2.imshow('Skladowa Cb po transmisji', upsampled_Cb)
    cv2.imshow('Skladowa Cr po transmisji', upsampled_Cr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

while True:
    print("\nMenu:")
    print("1. Zadanie 1")
    print("2. Zadanie 2")
    print("3. Zadanie 3")
    print("4. Zadanie 4")
    print("0. Wyjście")

    choice = input("Wybierz numer zadania (lub 0 aby wyjść): ")

    if choice == '1':
        task_1()
    elif choice == '2':
        task_2()
    elif choice == '3':
        task_3()
    elif choice == '4':
        task_4()
    elif choice == '0':
        print("Koniec programu.")
        break
    else:
        print("Niepoprawny wybór. Wybierz numer od 1 do 5 lub 0 aby wyjść.")
