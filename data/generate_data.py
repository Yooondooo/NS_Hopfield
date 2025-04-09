import os
import numpy as np
import cv2
import string

IMAGE_SIZE = 200
LETTERS = string.ascii_uppercase[:26]

def get_random_position(letter, font, font_scale, thickness, max_attempts=10):
    for _ in range(max_attempts):
        text_size, _ = cv2.getTextSize(letter, font, font_scale, thickness)
        text_width, text_height = text_size

        if text_width <= IMAGE_SIZE and text_height <= IMAGE_SIZE:
            max_x = IMAGE_SIZE - text_width
            max_y = IMAGE_SIZE - text_height
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(text_height, IMAGE_SIZE + 1)
            return (x, y)

        # Уменьшаем масштаб, если не помещается
        font_scale *= 0.9

    # В крайнем случае — по центру
    return (IMAGE_SIZE // 4, IMAGE_SIZE // 2)
def generate_letter_image(letter, noise_level=0):
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 8
    thickness = 10
    org = get_random_position(letter, font, font_scale, thickness)
    cv2.putText(img, letter, org, font, font_scale, (0), thickness, cv2.LINE_AA)
    if noise_level > 0:
        noise = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask = np.random.rand(IMAGE_SIZE, IMAGE_SIZE) < noise_level
        img[mask] = noise[mask]
    return img

def generate_letter_image2(letter, noise_level=0):
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, letter, (30, 190), font, 4, (0), 10, cv2.LINE_4)  # Увеличено шрифт и положение
    if noise_level > 0:
        noise = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask = np.random.rand(IMAGE_SIZE, IMAGE_SIZE) < noise_level
        img[mask] = noise[mask]
    return img

def generate_letter_image3(letter, noise_level=0):
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, letter, (15, 90), font, 4, (0), 10, cv2.LMEDS)  # Увеличено шрифт и положение
    if noise_level > 0:
        noise = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask = np.random.rand(IMAGE_SIZE, IMAGE_SIZE) < noise_level
        img[mask] = noise[mask]
    return img

def save_dataset(num = 3,noise = 0.1, dir = 'data'):
    X, y = [], []
    os.makedirs(f'{dir}', exist_ok=True)
    for j in range(0, num):
        os.makedirs(f'{dir}1{j}', exist_ok=True)
        for i, letter in enumerate(LETTERS):
            img = generate_letter_image(letter,noise * j)
            cv2.imwrite(f'{dir}1{j}/{letter}{j}.png', img)
            X.append(img)
            y.append(i)
            print("create 1")
    np.save(f'{dir}/X.npy', np.array(X))
    np.save(f'{dir}/y.npy', np.array(y))
    print(f"Dataset сохранен в data")

if __name__ == '__main__':
    num = 1
    save_dataset(num)
    #save_dataset(num,0.2,'data_test')


