from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
import math
import random
from tensorflow.keras import layers, models
import os

def load_data(data_dir):
    image_paths = []
    
    # Gremo skozi vse datoteke v mapi
    for filename in os.listdir(data_dir):
        # Preverimo, ce je datoteka .png
        if filename.endswith('.png') or filename.endswith('jpg') or filename.endswith('mp4'):
            # Kreiramo pot do slike in dodamo v tabelo
            image_path = os.path.join(data_dir, filename)
            image_paths.append(image_path)            

    return image_paths

def get_images_from_path(negative_image_paths, number_of_elements):
    augmented_negative_images = []
    
    for image_path in negative_image_paths:
        if len(augmented_negative_images) >= number_of_elements:
            break
        
        image = Image.open(image_path)
        augmented_negative_images.append(image)
    
    return augmented_negative_images

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

def preprocess(frame, target_size):
    # Spremenimo velikost slike
    image = frame.resize(target_size)
    
    # Pretvorimo sliko v NumPy tabelo
    image_array = np.array(image)
    
    # Pretvorimo sliko v sivinske vrednosti
    gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
    
    # Odstranimo sum iz slike
    denoised_image = gaussian_filter(gray, 1.5)
    
    # Lineariziramo sivinske vrednosti
    linearized_image = linearize_grayscale(denoised_image, 1.5)
            
    return linearized_image

# Preprocesiranje slik oz. framov preden gredo v augmentacijo
def preprocess_frames(frames, target_size):
    preprocessed_images = []
    
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # Ce je slika iz videa oz. frame (NumPy array)
            image = Image.fromarray(frame)
            normalized_image = preprocess(image, target_size)
            preprocessed_images.append(normalized_image)

        else:
            # Ce je slika iz prijave (image path)
            image = Image.open(frame)
            normalized_image = preprocess(image, target_size)
            preprocessed_images.append(normalized_image)

    return np.array(preprocessed_images)

#====================#
# Nakljucna rotacija #
#====================#
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
    height, width = image.shape
    output_height, output_width = dsize

    # Create an output image filled with zeros
    output_image = np.zeros((output_height, output_width), dtype=image.dtype)

    # Apply the transformation to each pixel in the output image
    for out_y in range(output_height):
        for out_x in range(output_width):
            # Map the output pixel coordinates to the input pixel coordinates
            in_x = M[0][0] * out_x + M[0][1] * out_y + M[0][2]
            in_y = M[1][0] * out_x + M[1][1] * out_y + M[1][2]

            # Reflect the image at the borders to avoid black edges
            x1, y1 = handle_border_reflect(in_x, in_y, width, height)
            if 0 <= x1 < width and 0 <= y1 < height:
                output_image[out_y, out_x] = image[y1, x1]

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
    
def random_rotation(image, max_angle=1):
    # Pretvorimo kot iz radianov v stopinje za OpenCV
    angle = np.random.uniform(-max_angle, max_angle) * (180.0 / np.pi)
    
    # Dobimo vrednosti
    height, width = image.shape

    rotation_matrix = getRotationMatrix2D(round(width / 2), round(height / 2), round(angle), 1.0)
    
    # Izvedemo rotacijo
    rotated_image = warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

#====================#
# Nakljucna svetlost #
#====================#
def random_value(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)

def adjust_brightness(image, delta):
    brightened_image = image + delta * 255
    brightened_image = clip_pixel_values(brightened_image)
    return brightened_image

def clip_pixel_values(brightened_image):
    clipped_image = []
    for row in brightened_image:
        clipped_row = []
        for pixel in row:
            clipped_pixel = max(0, min(255, pixel))
            clipped_row.append(clipped_pixel)
        clipped_image.append(clipped_row)
    return clipped_image


def random_brightness(image, max_delta=0.6):
    # Dolocimo svetlost za nakljucni faktor na intervalu [-max_delta, max_delta]
    delta = random_value(-max_delta, max_delta)
    brightened_image = adjust_brightness(image, delta)
    
    return brightened_image


#==================#
# Nakljucni premik #
#==================#
def random_translation(image, max_dx=0.2, max_dy=0.2):
    height, width = image.shape
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

#===================================#
# Nakljucna horizontalna preslikava #
#===================================#
def flip_image_horizontally(image):
    flipped_image = []
    for row in image:
        flipped_row = row[::-1]
        flipped_image.append(flipped_row)
    return flipped_image

def random_flip_horizontal(image):
    # Nakljucno izberemo ali obrnemo sliko
    if random.random() > 0.5:
        image = flip_image_horizontally(image)
    
    return image

augmentations = [
    random_rotation,
    random_brightness,
    random_translation, 
    random_flip_horizontal
]

def augment_images(images):
    augmented_images = []
    for image in images:        
        # Select a random augmentation for each image
        augmentation = random.choice(augmentations)
        
        # Dodamo augmentacijo
        augmented_image = augmentation(image)
        
        # Select a random augmentation for each image
        augmentation = random.choice(augmentations)
        
        # Dodamo augmentacijo
        augmented_image = augmentation(image)
        
        # Dodamo augmentirano sliko v tabelo
        augmented_images.append(augmented_image)
        
    return np.array(augmented_images)
    
def cut_videos(videoPaths):
    frames = []

    for video_path in videoPaths:
        # Zajamemo video
        cap = cv.VideoCapture(video_path)
        
        while True:
            # Beremo vsako sliko oz. frame
            ret, frame = cap.read()
            
            # Ce je slika zadnja ali napaka
            if not ret:
                break
            
            # Dodamo v tabelo
            frames.append(frame)
        
        # Sprostimo objekt
        cap.release()
    
    # Pretvorimo v np tabelo
    return np.array(frames)

#======================================#
# Izdelava modela za prepoznavo obraza #
#======================================#
def buildModel(input_shape):
    inputs = layers.Input(shape=input_shape) # Definicija vhoda za model

    # Konvolucijske plasti
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Gosto povezane plasti
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Izhodna plast (sigmoidna za binarno klasifikacijo)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Definiramo model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def createModel(videoPath, userId, directory_negative_images):    
    videos = load_data(videoPath)
    frames = cut_videos(videos)
    
    number_of_elements = len(frames) * 2
    negative_image_paths = load_data(directory_negative_images)
    augmented_negative_images = get_images_from_path(negative_image_paths, number_of_elements)
    
    target_size = (64, 64)

    preprocessed_positive_images = preprocess_frames(frames, target_size)
    augmented_positive_images = augment_images(preprocessed_positive_images)
    
    positive_images = np.concatenate((preprocessed_positive_images, augmented_positive_images), axis=0)  
      
    # Združimo pozitivne in negativne slike v podatke za treniranje
    # Pozitivne slike imajo label 1, negativne slike imajo label 0
    X_train = np.concatenate((positive_images, augmented_negative_images), axis=0)
    y_train = np.concatenate((np.ones(len(positive_images)), np.zeros(len(augmented_negative_images))))

    # Model build
    input_shape = (target_size[0], target_size[1], 1)
    model = buildModel(input_shape)

    # Compile model
    # Uporabljamo binary_crossentropy ker lahko slika samo pripada uporabniku ali ne
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Shranjevanje modela
    model.save(f'./files/{userId}/{userId}_model.h5')
    pass

def identifyFace(imagePath, userId):
    target_size = (64, 64)

    preprocessed_image = preprocess_frames([imagePath], target_size)
    
    model = tf.keras.models.load_model(f'./files/{userId}/{userId}_model.h5')

    prediction = model.predict(preprocessed_image)[0][0]


    if prediction > 0.5:
        return True
    else:
        return False