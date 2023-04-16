from glob import glob 
import cv2
from imgaug import augmenters as ia

def main():
    dataset_dir = 'raw_dataset/PURPLE_CHLORIS'

    images = glob('{}/*.jpeg'.format(dataset_dir))

    available_length = len(images)
    target_length = 150
    required_length = target_length - available_length

    print(images)

    raw_images = list()

    for image in images:
        image = cv2.imread(image)
        raw_images.append(image)

        # cv2.imshow('Dataset_images', image)
        # cv2.waitKey(0)


        # cv2.destroyAllWindows()

    augmenter = ia.Sequential([
        # ia.Rotate((-50, 50)),

        ia.Fliplr(0.5),
        ia.Flipud(0.5),

        ia.Affine(translate_percent = {'x':(-0.1, 0.1), 'y' : (-0.1, 0.1)},
                # rotate=(-30, 30),
                scale=(0.5, 1.5)),

        ia.Multiply((0.8, 1.2)),

        ia.LinearContrast((0.6, 1.4)),

        ia.GaussianBlur((0, 3)),

    ])

    augmented_images = list()

    while required_length > 0:

        if required_length > available_length:

            temp_augmented_images = augmenter(images = raw_images)
            augmented_images += temp_augmented_images

        else:
            temp_augmented_images = augmenter(images = raw_images[0:required_length])
            augmented_images += temp_augmented_images

        required_length -= available_length

    for idx, img in enumerate(augmented_images):

        # cv2.imshow('Augmented image', img)
        # cv2.waitKey(0)

        cv2.imwrite('{}/PURPLE_CHLORIS_{}.jpeg'.format(dataset_dir, idx+1), img)

if __name__ == '__main__' : 
    main()