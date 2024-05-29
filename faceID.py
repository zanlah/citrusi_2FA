from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
import math

# Preprocesiranje slik predn grejo v augmentacijo
def preprocess_images(image_paths, target_size = (64, 64)):
    preprocessed_images = []
    
    for image_path in image_paths:
        # Nalozi sliko
        image = Image.open(image_path)
        
        # Spremeni velikost v prej doloceno
        image = image.resize(target_size)
        
        # Pretvorimo v numpy tabelo
        image_array = np.array(image)
        
        # Normaliziramo vrednosti v intervalu [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Dodamo pre procesirano sliko v tabelo
        preprocessed_images.append(image_array)
        
    # Pretvorimo tabelo v numpy tabelo
    preprocessed_images = np.array(preprocessed_images)
    
    return preprocessed_images

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
    for _ in range(height):
        row = []
        for _ in range(width):
            pixel = [0] * channels
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
    
    return rotated_images

def random_brightness(images, max_delta=0.6):
    def adjust_brightness(image):
        # Dolocimo svetlost za nakljucni faktor na intervalu [-max_delta, max_delta]
        delta = np.random.uniform(-max_delta, max_delta)
        brightened_image = tf.image.adjust_brightness(image, delta)
        # Vrednosti nastavimo na interval [0, 1]
        brightened_image = tf.clip_by_value(brightened_image, 0.0, 1.0)
        return brightened_image
    
    # Posodobimo svetlost za vsako sliko
    brightened_images = [adjust_brightness(image) for image in images]
    
    # Pretvorimo tabelo slik v numpy tabelo
    brightened_images = np.array([img.numpy() for img in brightened_images])
    
    return brightened_images

def random_translation(images, max_dx=0.2, max_dy=0.2):
    def translate_image(image):
        height, width = image.shape[:2]
        
        # Izracunamo maksimalno stevilo pikslov za premik
        max_dx_pixels = int(max_dx * width)
        max_dy_pixels = int(max_dy * height)
        
        # Nakljucno izberemo stevilo pikslov za premik v x in y smeri
        tx = np.random.randint(-max_dx_pixels, max_dx_pixels + 1)
        ty = np.random.randint(-max_dy_pixels, max_dy_pixels + 1)
        
        # Naredimo matriko premika
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Uporabimo matriko in naredimo premik
        translated_image = cv.warpAffine(image, translation_matrix, (width, height), borderMode=cv.BORDER_REFLECT_101)
        return translated_image
    
    # Naredimo premik sliki
    translated_images = [translate_image(image) for image in images]
    
    return np.array(translated_images)

def random_flip_horizontal(images):
    def flip_image(image):
        # Nakljucno izberemo ali obrnemo sliko
        if np.random.rand() > 0.5:
            flipped_image = tf.image.flip_left_right(image)
        else:
            flipped_image = image
        return flipped_image
    
    # Obrnemo sliko
    flipped_images = [flip_image(image) for image in images]
    
    # Pretvorimo tabelo v numpy tabelo
    flipped_images = np.array([img for img in flipped_images])
    
    return flipped_images

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
    return np.array(images)

def createModel(videoPath, userId):
    pass

def identifyFace(imagePath, userId):
    pass