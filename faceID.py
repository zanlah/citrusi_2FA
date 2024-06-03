from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
import math
import random

def convolution(image, kernel):
    # Definiramo dimenzije slike in jedra
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape
    
    # Nastavimo padding
    padding_rows = kernel_rows // 2
    padding_cols = kernel_cols // 2
    
    # Paddamo sliko z robnimi vrednostmi
    padded_image = np.pad(image, ((padding_rows, padding_rows), (padding_cols, padding_cols)), mode='edge')
    
    # Inicializiramo tabelo v kateri bo koncni rezultat
    convolution_result = [[0] * image_cols for _ in range(image_rows)]
    
    # Konvolucija
    for i in range(image_rows):
        for j in range(image_cols):
            # Dobimo obmocje interesa iz paddane slike
            region_of_interest = padded_image[i:i+kernel_rows, j:j+kernel_cols]
            # Poskrbimo, da je obmocje interesa enak kot jedro
            if region_of_interest.shape == kernel.shape:
                convolution_result[i][j] = np.sum(region_of_interest * kernel)
                    
    # Poskrbimo, da so vrednosti med 0-255 za vsak kanal
    convolution_result = np.clip(convolution_result, 0, 255).astype(np.uint8)
                    
    return convolution_result

def gaussian_filter(image, sigma):
    # Velikost jedra
    kernel_size = int((2 * sigma) * 2 + 1)
    
    # Ustvarimo prazno jedro
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Izracunamo jedro glede na gaussovo funkcijo
    k = kernel_size // 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            kernel[i+k][j+k] = 1 / (2 * math.pi * sigma**2) * math.exp(-((i)**2 + (j)**2) / (2 * sigma**2))
    
    # Normalizacija jedra
    kernel /= np.sum(kernel)
    
    # Uporaba funkcije konvolucije
    gaussian_image = convolution(image, kernel)
    
    # Normalizacija rezultata
    gaussian_image = (gaussian_image - np.min(gaussian_image)) / (np.max(gaussian_image) - np.min(gaussian_image)) * 255
    
    return gaussian_image.astype(np.uint8)

def linearize_grayscale(image, gamma):
    # Uporabimo gamma korekcijo za linearizacijo sivinskih vrednosti
    table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype('uint8')
    
    # Uporabimo funkicjo za aplikacijo gamma korekcije na sliko s tabelo
    linearized = cv.LUT(image, table)
    
    return linearized

# Preprocesiranje slik predn grejo v augmentacijo
def preprocess_images(image_paths, target_size):
    preprocessed_images = []
    
    for image_path in image_paths:
        # Nalozimo sliko
        image = Image.open(image_path)
        
        # Spremenimo velikost slike
        image = image.resize(target_size)
        
        # Pretvorimo sliko v NumPy tabelo
        image_array = np.array(image)
        
        # Pretvorimo sliko v sivinske vrednosti
        gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        
        # Odstranimo sum iz slike
        denoised_image = gaussian_filter(gray, 2.2)
        
        # Lineariziramo sivinske vrednosti
        linearized_image = linearize_grayscale(denoised_image, 2.2)
        
        # Normaliziramo slikovne pike na interval [0, 1]
        normalized_image = linearized_image.astype('float32') / 255.0
        
        # Dodamo predprocesirano sliko v seznam
        preprocessed_images.append(normalized_image)
    
    return np.array(preprocessed_images)

# ===================#
# Nakljucna rotacija #
# ===================#
def getRotationMatrix2D(center_x, center_y, angle, scale=1.0):
    # Pretvorimo kot iz radianov v stopinje
    angle_degrees = angle * 180.0 / math.pi
    
    # Kalkuliramo sinus in kosinus kota
    alpha = math.cos(angle_degrees * math.pi / 180.0)
    beta = math.sin(angle_degrees * math.pi / 180.0)
    
    # Konstruiramo rotacijsko matriko
    rotation_matrix = [
        [alpha * scale, beta * scale, (1 - alpha) * center_x - beta * center_y],
        [-beta * scale, alpha * scale, beta * center_x + (1 - alpha) * center_y]
    ]
    
    return rotation_matrix

def warpAffine(image, M, dsize):
    height, width, channels = image.shape
    output_height, output_width = dsize

    # Kreiramo sliko za izhod z samimi niclami
    output_image = []
    for _ in range(output_height):
        row = []
        for _ in range(output_width):
            pixel = [0.0] * channels
            row.append(pixel)
        output_image.append(row)

    # Damo transformacijo na vsak piksel v izhodni sliki
    for out_y in range(output_height):
        for out_x in range(output_width):
            # Mapiramo koordinate izhodnega piksla na koordinate piksla vhodne slike
            in_x = M[0][0] * out_x + M[0][1] * out_y + M[0][2]
            in_y = M[1][0] * out_x + M[1][1] * out_y + M[1][2]

            # Zrcalimo sliko, da na robovih ni crno
            x1, y1 = handle_border_reflect(in_x, in_y, width, height)
            if 0 <= x1 < width and 0 <= y1 < height:
                output_image[out_y][out_x] = image[y1][x1][:]

    return output_image

def handle_border_reflect(x, y, width, height):
    x = int(x)
    y = int(y)

    # Zrcaljenje x koordinat
    if x < 0:
        x = -x - 1
    elif x >= width:
        x = 2 * width - x - 1

    # Zrcaljenje y koordinat
    if y < 0:
        y = -y - 1
    elif y >= height:
        y = 2 * height - y - 1

    return x, y
    
def random_rotation(images, max_angle=1):
    def rotate_and_fill(image):
        # Pretvorimo kot iz radianov v stopinje za OpenCV
        angle = np.random.uniform(-max_angle, max_angle) * (180.0 / np.pi)

        # Rotiramo sliko
        height = image.shape[0]
        width = image.shape[1]
     
        rotation_matrix = getRotationMatrix2D(round(width / 2), round(height / 2), round(angle), 1.0)

        # Izvedemo rotacijo
        rotated_image = warpAffine(image, rotation_matrix, (width, height))

        return rotated_image

    # Rotiramo in zapolnimo prazne dele slike po rotaciji
    rotated_images = [rotate_and_fill(image) for image in images]
    
    return np.array(rotated_images)

#====================#
# Nakljucna svetlost #
#====================#
def random_value(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)

def adjust_brightness(image, delta):
    brightened_image = []
    for row in image:
        brightened_row = []
        for pixel in row:
            adjusted_pixel = [channel + delta for channel in pixel]
            adjusted_pixel = [max(0.0, min(1.0, channel)) for channel in adjusted_pixel]
            brightened_row.append(adjusted_pixel)
        brightened_image.append(brightened_row)
    return brightened_image

def clip_pixel_values(brightened_image):
    clipped_image = []
    for row in brightened_image:
        clipped_row = []
        for pixel in row:
            clipped_pixel = []
            for channel in pixel:
                clipped_channel = max(0.0, min(1.0, channel))
                clipped_pixel.append(clipped_channel)
            clipped_row.append(clipped_pixel)
        clipped_image.append(clipped_row)
    return clipped_image

def random_brightness(images, max_delta=0.6):
    def change_brightness(image):   
        # Dolocimo svetlost za nakljucni faktor na intervalu [-max_delta, max_delta]
        delta = random_value(-max_delta, max_delta)
        brightened_image = adjust_brightness(image, delta)
        # Vrednosti nastavimo na interval [0, 1]
        brightened_image = clip_pixel_values(brightened_image)
        return brightened_image
    
    # Posodobimo svetlost za vsako sliko
    brightened_images = [change_brightness(image) for image in images]
        
    return np.array(brightened_images)

#==================#
# Nakljucni premik #
#==================#
def random_translation(images, max_dx=0.2, max_dy=0.2):
    def translate_image(image):
        height, width, channels = image.shape
        # Izracunamo maksimalno stevilo pikslov za premik
        max_dx_pixels = int(max_dx * width)
        max_dy_pixels = int(max_dy * height)
        
        # Nakljucno izberemo stevilo pikslov za premik v x in y smeri
        tx = random.randrange(-max_dx_pixels, max_dx_pixels + 1)
        ty = random.randrange(-max_dy_pixels, max_dy_pixels + 1)
        
        # Naredimo matriko premika
        translation_matrix = [[1, 0, tx], [0, 1, ty]]
        
        # Uporabimo matriko in naredimo premik
        translated_image = warpAffine(image, translation_matrix, (width, height))
        
        return translated_image
    
    # Naredimo premik sliki
    translated_images = [translate_image(image) for image in images]

    return np.array(translated_images)

#===================================#
# Nakljucna horizontalna preslikava #
#===================================#
def flip_image_horizontally(image):
    flipped_image = []
    for row in image:
        flipped_row = row[::-1]
        flipped_image.append(flipped_row)
    return flipped_image

def random_flip_horizontal(images):
    def flip_image(image):
        # Nakljucno izberemo ali obrnemo sliko
        if random.random() > 0.5:
            flipped_image = flip_image_horizontally(image)
        else:
            flipped_image = image
        return flipped_image
    
    # Obrnemo sliko
    flipped_images = [flip_image(image) for image in images]
        
    return np.array(flipped_images)

augmentations = [
    random_rotation,
    random_brightness,
    random_translation,
    random_flip_horizontal
]

def augment_images(images):
    for augmentation in augmentations:
        if callable(augmentation):
            images = augmentation(images)
        else:
            images = augmentation(images)
    
    augmented_images = []
    for image_array in images:
        # Pretvorimo sliko nazaj na interval [0, 255]
        image_array = (image_array * 255).astype('uint8')
        
        # Pretvorimo numpy tabelo v PIL sliko
        image = Image.fromarray(image_array)
        augmented_images.append(image)

    return augmented_images


#======================================#
# Izdelava modela za prepoznavo obraza #
#======================================#
def createModel(videoPath, userId):
    # Najprej klices funkcijo preprocess_images. V argumente das poti do vseh slik npr. array: [faceid_images\0.png, faceid_images\1.png] in velikost v kero sliko pretvorimo npr. (64, 64)
    # Pol se klice augment_images, tu nt pol das ka se returna od preprocess_images
    pass

def identifyFace(imagePath, userId):
    pass
