import cv2
import numpy as np
import os
from PIL import Image
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

def save_ppm_p3(image, filename):
    with open(filename, 'w') as f:
        height, width, _ = image.shape
        f.write(f'P3\n{width} {height}\n255\n')
        for row in image:
            for pixel in row:
                f.write(f'{pixel[2]} {pixel[1]} {pixel[0]} ')
            f.write('\n')

def save_ppm_p6(image, filename):
    with open(filename, 'wb') as f:
        height, width, _ = image.shape
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(image[:,:,::-1].tobytes())

def read_ppm_p3(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    assert lines[0].strip() == 'P3'
    width, height = map(int, lines[1].strip().split())
    max_val = int(lines[2].strip())
    assert max_val == 255
    image_data = [int(value) for line in lines[3:] for value in line.split()]
    image = np.array(image_data, dtype=np.uint8).reshape((height, width, 3))
    return image[:, :, ::-1]

def read_ppm_p6(filename):
    with open(filename, 'rb') as f:
        header = f.readline() + f.readline() + f.readline()
        width, height = map(int, header.split()[1:3])
        image_data = np.fromfile(f, dtype=np.uint8)
    image = image_data.reshape((height, width, 3))
    return image[:, :, ::-1]

def task_2_1():
    print("Wybrałeś zadanie 2.1.")

    # Generowanie sztucznego obrazu
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75, 2] = 255  # Czerwony kwadrat
    image[:, :, 1] = 128  # Zielone tło

    # Zapisanie obrazu w formatach P3 i P6
    save_ppm_p3(image, 'image_p3.ppm')
    save_ppm_p6(image, 'image_p6.ppm')

    # Odczytanie obrazu z formatów P3 i P6
    image_from_p3 = read_ppm_p3('image_p3.ppm')
    image_from_p6 = read_ppm_p6('image_p6.ppm')

    # Porównanie rozmiarów plików
    size_p3 = os.path.getsize('image_p3.ppm')
    size_p6 = os.path.getsize('image_p6.ppm')
    print(f'Rozmiar pliku P3: {size_p3} bajtów')
    print(f'Rozmiar pliku P6: {size_p6} bajtów')


def task_2_2():
    print("Wybrałeś zadanie 2.2.")

    # Wymiary obrazu
    height = 100
    width = 800  # Szerokość dostosowana do liczby przejść

    # Tworzenie pustego obrazu
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Definiowanie kolorów przejść
    colors = [
        (0, 0, 0),       # Czarny
        (255, 0, 0),     # Niebieski
        (255, 255, 0),   # Błękitny
        (0, 255, 0),     # Zielony
        (0, 255, 255),   # Żółty
        (0, 0, 255),     # Czerwony
        (255, 0, 255),   # Różowy
        (255, 255, 255), # Biały
    ]

    # Liczba przejść
    num_transitions = len(colors) - 1

    # Generowanie przejść kolorów
    for i in range(num_transitions):
        start_color = np.array(colors[i])
        end_color = np.array(colors[i+1])
        for j in range(width // num_transitions):
            alpha = j / (width // num_transitions)
            color = (1 - alpha) * start_color + alpha * end_color
            image[:, i * (width // num_transitions) + j, :] = color

    # Zapisywanie obrazu w formacie P6
    save_ppm_p6(image, 'rgb_color_transition.ppm')
    print("Obraz przejścia kolorów RGB został wygenerowany i zapisany.")



def task_2_3():
    print("Wybrałeś zadanie 2.3.")

    # Otwarcie obrazu PPM
    ppm_image = Image.open('rgb_color_transition.ppm')

    # Zapisanie obrazu w formacie PNG
    ppm_image.save('rgb_color_transition.png', 'PNG')

    print("Obraz został zapisany w formacie PNG.")

while True:
    print("\nMenu:")
    print("1. Zadanie 1")
    print("2. Zadanie 2")
    print("3. Zadanie 3")
    print("4. Zadanie 4")
    print("6. Zadanie 1 Lista 2")
    print("7. Zadanie 2 Lista 2")
    print("8. Zadanie 3 Lista 2")
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
    elif choice == '6':
        task_2_1()
    elif choice == '7':
        task_2_2()
    elif choice == '8':
        task_2_3()
    elif choice == '0':
        print("Koniec programu.")
        break
    else:
        print("Niepoprawny wybór. Wybierz numer od 1 do 5 lub 0 aby wyjść.")
