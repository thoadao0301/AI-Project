from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import cv2
import os
import sys


def main(args):
    list_sub_dir = os.listdir(args.data_dir)
    datagen1 = ImageDataGenerator(
        rotation_range=args.rotation_range1,
        width_shift_range=args.width_shift_range1,
        height_shift_range=args.height_shift_range1,
        channel_shift_range=args.channel_shift_range,
        horizontal_flip=args.horizontal_flip)
    datagen2 = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=args.rotation_range2,
        width_shift_range=args.width_shift_range2,
        height_shift_range=args.height_shift_range2,
        brightness_range=args.brightness_range,
        horizontal_flip=args.horizontal_flip)
    for sub_dir in list_sub_dir:
        sub_dir_path = os.path.join(args.data_dir,sub_dir)
        list_image = os.listdir(sub_dir_path)
        for image_name in list_image:
            image_path = os.path.join(sub_dir_path,image_name)
            img = cv2.imread(image_path)
            datagen1.fit(np.expand_dims(img, axis=0))
            datagen2.fit(np.expand_dims(img,axis=0))
            for num in range(5):
                image1 = datagen1.flow(np.expand_dims(img, axis=0))
                image2 = datagen2.flow(np.expand_dims(img, axis=0))
                cv2.imwrite(f'{sub_dir_path}/data_generate{image_name}{num+1}.jpg',image1[0][0])
                cv2.imwrite(f'{sub_dir_path}/data_generate{image_name}{num+6}.jpg',image2[0][0])
            print('Successful create 10 images in folder:',sub_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory')

    parser.add_argument('--rotation_range1', type=float,
                        help='', default=20)
    parser.add_argument('--rotation_range2', type=float,
                        help='', default=30)
    parser.add_argument('--width_shift_range1', type=float,
                        help='', default=0.2)
    parser.add_argument('--width_shift_range2', type=float,
                        help='', default=0.1)
    parser.add_argument('--height_shift_range1', type=float,
                        help='', default=0.2)
    parser.add_argument('--height_shift_range2', type=float,
                        help='', default=0.1)
    parser.add_argument('--brightness_range', type=float,
                        help='', default=(1.3,1.6))
    parser.add_argument('--channel_shift_range', type=float,
                        help='', default=50)
    parser.add_argument('--horizontal_flip', type=bool,
                        help='', default=False)



    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



