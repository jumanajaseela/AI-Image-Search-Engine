import cv2


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded")

    image = cv2.resize(image, (224, 224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ == "__main__":
    path = "val2017/val2017/000000000139.jpg"

    image = preprocess_image(path)

    print("Image processed successfully")
    print("Shape:", image.shape)
