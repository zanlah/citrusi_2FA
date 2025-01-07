from mpi4py import MPI
from PIL import Image
import numpy as np
import cv2 as cv
import os
import socket
import random

# Pomozne funkcije
# --------------------------------------------------------------------------------

def load_data(data_dir):
    print(f"Nalaganje slik iz: {data_dir}")
    image_paths = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4')):
            image_path = os.path.join(data_dir, filename)
            image_paths.append(image_path)
    return image_paths

def get_images_from_path(negative_image_paths, number_of_elements):
    """
    Pridobivanje dolocenega stevila sivinskih slik iz mape z negativnimi slikami
    """
    images = []
    for image_path in negative_image_paths:
        if len(images) >= number_of_elements:
            break
        image = Image.open(image_path).convert('L')  # Pretvorba v sivinsko sliko
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
    """
    Pridobivanje slik iz videa pri registraciji
    """
    frames = []
    for video_path in video_paths:
        cap = cv.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    print(f"Stevilo vseh dobljenih slik: {len(frames)}")
    return np.array(frames)


# Funkcije za augmentacijo
# --------------------------------------------------------------------------------

def random_rotation(image, max_angle=1):
    angle = np.random.uniform(-max_angle, max_angle) * (180.0 / np.pi)
    height, width = image.shape
    M = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv.warpAffine(image, M, (width, height), borderMode=cv.BORDER_REFLECT)

def random_brightness(image, max_delta=0.6):
    delta = random.uniform(-max_delta, max_delta)
    return np.clip(image + delta * 255, 0, 255).astype(np.uint8)

def random_translation(image, max_dx=0.2, max_dy=0.2):
    height, width = image.shape
    tx, ty = random.randint(-int(max_dx * width), int(max_dx * width)), random.randint(-int(max_dy * height), int(max_dy * height))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv.warpAffine(image, M, (width, height), borderMode=cv.BORDER_REFLECT)

def random_flip_horizontal(image):
    image = np.squeeze(image)
    return cv.flip(image, 1) if random.random() > 0.5 else image

augmentations = [
    random_rotation,
    random_brightness,
    random_translation,
    random_flip_horizontal
]

def augment_images_parallel(images):
    """
    Porazdeljeno augmentiranje slik. Vsak proces dobi "chunk" slik. Na koncu so vse slike zbrane nazaj v proces 0
    Nad sliko se kljice nakljucna augmentacija
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Izracun za deljenje slik med procesi
    chunk_size = len(images) // size
    remainder = len(images) % size

    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    if rank == size - 1:
        end_idx += remainder  # zadnji proces obdela mozni ostanek

    local_chunk = images[start_idx:end_idx]
    host = socket.gethostname()
    print(f"Proces {rank} na  {host} obdeluje {len(local_chunk)} slik.")

    local_augmented = []
    for i, img in enumerate(local_chunk):
        augmentation = random.choice(augmentations)
        augmented_image = augmentation(img)
        local_augmented.append(augmented_image)

    local_augmented = np.array(local_augmented)

    # Zbiranje rezultatov v procesu 0
    if rank == 0:
        counts = [chunk_size] * size
        counts[-1] += remainder
        
        h, w = images.shape[1], images.shape[2]
        counts = [count * h * w for count in counts]
        
        displacements = [sum(counts[:i]) for i in range(size)]
        
        all_augmented = np.empty((len(images), h, w), dtype=np.uint8)
    else:
        counts = None
        displacements = None
        all_augmented = None

    local_augmented_flat = local_augmented.reshape(local_augmented.shape[0], -1)

    local_augmented_flat_1d = local_augmented_flat.ravel()

    # Gather vse slike na procesu 0
    comm.Gatherv(
        sendbuf=local_augmented_flat_1d,
        recvbuf=(all_augmented, counts, displacements, MPI.UNSIGNED_CHAR),
        root=0
    )

    if rank == 0:
        print("Augmentacija koncana...")
        return all_augmented
    else:
        return None


# "create_model" Funkcija opravi samo augmentacijo
# --------------------------------------------------------------------------------

def create_model(video_path, user_id, negative_image_dir):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("1) Nalaganje videa...")
        videos = load_data(video_path)

        print("2) Kreiranje slik iz videa...")
        frames = cut_videos(videos)

        number_of_elements = len(frames) * 2
        print("3) Nalaganje negativnih slik...")
        negative_image_paths = load_data(negative_image_dir)
        negative_images = get_images_from_path(negative_image_paths, number_of_elements)

        target_size = (64, 64)
        print("4) Procesiranje pozitivnih slik...")
        positive_images = preprocess_frames(frames, target_size)
    else:
        videos = None
        frames = None
        negative_images = None
        positive_images = None

    positive_images = comm.bcast(positive_images, root=0)
    augmented_positive_images = augment_images_parallel(positive_images)

    if rank == 0:
        all_positive_images = np.concatenate([positive_images, augmented_positive_images])

        output_dir = f'./files/{user_id}/augmented/'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Shranjevanje augmentiranih slik v mapo {output_dir} ...")
        for i, img in enumerate(augmented_positive_images):
            save_path = os.path.join(output_dir, f"augmented_{i}.png")
            Image.fromarray(img).save(save_path)

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("===== Zacetek augmentacije =====")
        user_id = '1'
        video_directory = f'./files/{user_id}/register/'
        negative_images_directory = './negative_images'
        print(f"User ID: {user_id}")
    else:
        user_id = None
        video_directory = None
        negative_images_directory = None

    create_model(
        video_directory if rank == 0 else None, 
        user_id if rank == 0 else None,
        negative_images_directory if rank == 0 else None
    )

    if rank == 0:
        print(f"Augmentacija koncana za user ID: {user_id}")
        print("===== Konec augmentacije =====")

if __name__ == "__main__":
    main()
