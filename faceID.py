import numpy as np
import cv2 as cv
import tensorflow as tf

def random_rotation(images, max_angle=1):
    def rotate_and_fill(image):
        # Pretvorimo kot iz radianov v stopinje za OpenCV
        angle = np.random.uniform(-max_angle, max_angle) * (180.0 / np.pi)
        
        # Rotiramo sliko
        height, width = image.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        
        # Izvedemo rotacijo
        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height), borderMode=cv.BORDER_REFLECT_101)
        return rotated_image

    # Rotiramo in zapolnimo prazne dele slike po rotaciji
    rotated_images = [rotate_and_fill(image) for image in images]
    
    return np.array(rotated_images)

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
    return images

def createModel(videoPath, userId):
    pass

def identifyFace(imagePath, userId):
    pass