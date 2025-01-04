from mpi4py import MPI
from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

def load_data(data_dir):
    image_paths = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4')):
            image_paths.append(os.path.join(data_dir, filename))
    return image_paths

def get_images_from_path(negative_image_paths, number_of_elements):
    images = []
    for image_path in negative_image_paths:
        if len(images) >= number_of_elements:
            break
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        images.append(np.array(image))
    return images

def preprocess(frame, target_size):
    image = frame.resize(target_size)
    image_array = np.array(image)
    gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
    return gray

def preprocess_frames(frames, target_size):
    preprocessed_images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = Image.open(frame)
        preprocessed_images.append(preprocess(image, target_size))
    return np.array(preprocessed_images)

def cut_videos(video_paths):
    frames = []
    for video_path in video_paths:
        cap = cv.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return np.array(frames)

def augment_images_parallel(images):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Configure the data generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Calculate chunk size for each process
    chunk_size = len(images) // size
    remainder = len(images) % size
    
    # Adjust chunk size for the last process to handle remaining images
    if rank == size - 1:
        local_chunk = images[rank * chunk_size : rank * chunk_size + chunk_size + remainder]
    else:
        local_chunk = images[rank * chunk_size : (rank + 1) * chunk_size]
    
    # Process local chunk
    local_augmented = []
    for image in local_chunk:
        image = image.reshape((1, *image.shape, 1))  # Reshape for augmentation
        aug_iter = datagen.flow(image, batch_size=1)
        local_augmented.append(aug_iter[0].astype(np.uint8).squeeze())
    
    # Convert to numpy array
    local_augmented = np.array(local_augmented)
    
    # Gather results from all processes
    if rank == 0:
        # Calculate receive counts and displacements for gathering
        counts = [chunk_size] * size
        counts[-1] += remainder  # Add remainder to last process
        counts = [count * images.shape[1] * images.shape[2] for count in counts]  # Adjust for image dimensions
        
        displacements = [sum(counts[:i]) for i in range(size)]
        
        # Initialize array to receive all augmented images
        all_augmented = np.empty((len(images), *images.shape[1:]), dtype=np.uint8)
    else:
        counts = None
        displacements = None
        all_augmented = None
    
    # Use numpy array directly with MPI
    comm.Gatherv(sendbuf=local_augmented,
                 recvbuf=(all_augmented, counts, displacements, MPI.UNSIGNED_CHAR),
                 root=0)
    
    return all_augmented if rank == 0 else None

def build_model(input_shape):
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

def create_model(video_path, user_id, negative_image_dir):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        videos = load_data(video_path)
        frames = cut_videos(videos)
        number_of_elements = len(frames) * 2
        negative_image_paths = load_data(negative_image_dir)
        negative_images = get_images_from_path(negative_image_paths, number_of_elements)

        target_size = (64, 64)
        positive_images = preprocess_frames(frames, target_size)
    else:
        positive_images = None
    
    # Broadcast positive_images to all processes
    positive_images = comm.bcast(positive_images, root=0)
    
    # Perform parallel augmentation
    augmented_positive_images = augment_images_parallel(positive_images)
    
    if rank == 0:
        positive_images = np.concatenate([positive_images, augmented_positive_images])
        X_train = np.concatenate((positive_images, negative_images))
        y_train = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))))

        input_shape = (target_size[0], target_size[1], 1)
        model = build_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Disable progress bar to avoid Unicode issues
        model.fit(
            X_train, 
            y_train, 
            epochs=10, 
            batch_size=32,
            verbose=2  # Use verbose=2 for less detailed output without progress bar
        )

        os.makedirs(f'./files/{user_id}', exist_ok=True)
        model.save(f'./files/{user_id}/{user_id}_model.h5')

def identify_face(image_path, user_id):
    target_size = (64, 64)
    preprocessed_image = preprocess_frames([image_path], target_size)
    preprocessed_image = preprocessed_image.reshape((1, 64, 64, 1))

    model = tf.keras.models.load_model(f'./files/{user_id}/{user_id}_model.h5')
    # Disable verbose output during prediction
    prediction = model.predict(preprocessed_image, verbose=0)[0][0]

    print(f"Confidence score: {prediction:.2f}")
    return prediction > 0.7

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        video_directory = f'./files/{1}/register/'
        negative_images_directory = './negative_images'
        user_id = '1'
        
        print("Creating model...")
    
    create_model(video_directory if rank == 0 else None, 
                user_id if rank == 0 else None,
                negative_images_directory if rank == 0 else None)
    
    if rank == 0:
        print(f"Model created and saved for user ID: {user_id}")
        
        test_image_path = f'./files/{1}/login/image.png'
        print(f"Identifying face for user ID: {user_id} in image: {test_image_path}")
        is_face_identified = identify_face(test_image_path, user_id)
        
        if is_face_identified:
            print("Face belongs to the user.")
        else:
            print("Face does NOT belong to the user.")

if __name__ == "__main__":
    main()
