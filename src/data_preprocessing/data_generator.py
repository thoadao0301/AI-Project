"""Generate data from image."""
#   MIT License
#  Copyright (c) 2021. TranPhuongNam,DaoLeBaoThoa,NguyenDiemUyenPhuong
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import cv2
import os
import sys
from tqdm import tqdm

def main(args):
    list_sub_dir = os.listdir(args.data_dir)
    datagen1 = ImageDataGenerator(
        featurewise_center=args.featurewise_center,
        samplewise_center=args.samplewise_center,
        featurewise_std_normalization=args.featurewise_std_normalization,
        samplewise_std_normalization=args.samplewise_std_normalization,
        zca_whitening=args.zca_whitening,
        zca_epsilon=args.zca_epsilon,
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        brightness_range=args.brightness_range,
        shear_range=args.shear_range,
        zoom_range=args.zoom_range,
        channel_shift_range=args.channel_shift_range,
        fill_mode=args.fill_mode,
        cval=args.cval,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        rescale=args.rescale,
        preprocessing_function=args.preprocessing_function,
        data_format=args.data_format,
        validation_split=args.validation_split,
        dtype=args.dtype)
    n=len(list_sub_dir)
    for i,sub_dir in enumerate(list_sub_dir):
        sub_dir_path = os.path.join(args.data_dir,sub_dir)
        list_image = os.listdir(sub_dir_path)
        print('Generate images folder: ',i,'/',n)
        with tqdm(total=len(list_image), file=sys.stdout) as pbar:
            for image_name in list_image:
                image_path = os.path.join(sub_dir_path,image_name)
                img = cv2.imread(image_path)
                datagen1.fit(np.expand_dims(img, axis=0))
                for num in range(5):
                    image1 = datagen1.flow(np.expand_dims(img, axis=0))
                    cv2.imwrite(f'{sub_dir_path}/data_generate{image_name}{num+1}.jpg',image1[0][0])
                pbar.update(1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='''Path to the data directory. The default generator are:         
        rotation_range=20
        width_shift_range=0.2
        height_shift_range=0.2
        channel_shift_range=50
        horizontal_flip=True''')

    parser.add_argument('--featurewise_center', type=bool,
                        help='Boolean. Set input mean to 0 over the dataset, feature-wise.',
                        default=False)
    parser.add_argument('--samplewise_center', type=bool,
                        help='Boolean. Set each sample mean to 0.',
                        default=False)
    parser.add_argument('--featurewise_std_normalization', type=bool,
                        help='Boolean. Divide inputs by std of the dataset, feature-wise.',
                        default=False)
    parser.add_argument('--samplewise_std_normalization', type=bool,
                        help='Boolean. Divide each input by its std.',
                        default=False)
    parser.add_argument('--zca_epsilon', type=float,
                        help='epsilon for ZCA whitening. Default is 1e-6.',
                        default=1e-06)
    parser.add_argument('--zca_whitening', type=bool,
                        help='Boolean. Apply ZCA whitening.',
                        default=False)
    parser.add_argument('--rotation_range', type=int,
                        help='Int. Degree range for random rotations. Default is 20.',
                        default=20)
    parser.add_argument('--width_shift_range', type=float,
                        help='''Float, 1-D array-like or int
* float: fraction of total width, if < 1, or pixels if >= 1.
* 1-D array-like: random elements from the array.
* int: integer number of pixels from interval (-width_shift_range, +width_shift_range)
* With width_shift_range=2 possible values are integers [-1, 0, +1], same as with width_shift_range=[-1, 0, +1], while with width_shift_range=1.0 possible values are floats in the interval [-1.0, +1.0).''',
                        default=0.2)
    parser.add_argument('--height_shift_range', type=float,
                        help='''	Float, 1-D array-like or int
* float: fraction of total height, if < 1, or pixels if >= 1.
* 1-D array-like: random elements from the array.
* int: integer number of pixels from interval (-height_shift_range, +height_shift_range)
* With height_shift_range=2 possible values are integers [-1, 0, +1], same as with height_shift_range=[-1, 0, +1], while with height_shift_range=1.0 possible values are floats in the interval [-1.0, +1.0).''',
                        default=0.2)
    parser.add_argument('--brightness_range',
                        help='Tuple or list of two floats. Range for picking a brightness shift value from. Default is (1.3,1.6).',
                        default=None)
    parser.add_argument('--shear_range', type=float,
                        help='Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)',
                        default=0.0)
    parser.add_argument('--zoom_range', type=float,
                        help='''Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].''',
                        default=0.0)
    parser.add_argument('--channel_shift_range', type=float,
                        help='Float. Range for random channel shifts. Default is 50',
                        default=50)
    parser.add_argument('--fill_mode', type=str,choices=["constant", "nearest", "reflect" or "wrap"],
                        help='''One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode:
* 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
* 'nearest': aaaaaaaa|abcd|dddddddd
* 'reflect': abcddcba|abcd|dcbaabcd
* 'wrap': abcdabcd|abcd|abcdabcd''',
                        default='nearest')
    parser.add_argument('--cval', type=float,
                        help='Float or Int. Value used for points outside the boundaries when fill_mode = "constant".',
                        default=0.0)
    parser.add_argument('--horizontal_flip', type=float,
                        help='Boolean. Randomly flip inputs horizontally. Default is True',
                        default=True)
    parser.add_argument('--vertical_flip', type=float,
                        help='Boolean. Randomly flip inputs vertically.',
                        default=False)
    parser.add_argument('--rescale', type=float,
                        help='	rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations).',
                        default=0)
    parser.add_argument('--preprocessing_function',
                        help='	function that will be applied on each input. The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.',
                        default=None)
    parser.add_argument('--data_format', type=float,
                        help='Image data format, either "channels_first" or "channels_last". "channels_last" mode means that the images should have shape (samples, height, width, channels), "channels_first" mode means that the images should have shape (samples, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".',
                        default=None)
    parser.add_argument('--validation_split', type=float,
                        help='Float. Fraction of images reserved for validation (strictly between 0 and 1).',
                        default=0.0)
    parser.add_argument('--dtype',
                        help='Dtype to use for the generated arrays.',
                        default=None)



    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



