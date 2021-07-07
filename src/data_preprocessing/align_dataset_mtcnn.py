"""Performs face alignment and stores face thumbnails in the output directory."""
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

import sys
import os
import argparse
import numpy as np
import random
import cv2
from time import sleep
from PIL import Image
from mtcnn import MTCNN
from tqdm import tqdm

def crop_faces_helper(img,face_bbox,margin):
    x1, y1, width, height = face_bbox

    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    x1 = x1 - margin / 2 if x1 - margin / 2 > 0 else 0
    y1 = y1 - margin / 2 if y1 - margin / 2 > 0 else 0
    x2 = x2 + margin / 2 if x2 + margin / 2 < img.shape[1] else img.shape[1]
    y2 = y2 + margin / 2 if y2 + margin / 2 < img.shape[0] else img.shape[0]
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    face = Image.fromarray(face)
    return face

def extract_faces(img_array, detector,detect_multiple_faces,threshold, image_size=160, margin=44):
    faces_list, bbox = [], []
    # convert channel

    results = detector.detect_faces(img_array)

    for face in results:
        confidence = face['confidence']
        if confidence < threshold:
            continue
        face_bbox = face['box']
        face = crop_faces_helper(img_array,face_bbox,margin)
        face = face.resize((image_size, image_size))
        face = np.asarray(face)
        faces_list.append(face)
        bbox.append(face_bbox)
        if not detect_multiple_faces:
            return faces_list, bbox

    return faces_list, bbox


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('Creating networks and loading parameters')
    detector = MTCNN()
    dataset = get_dataset(args.input_dir)
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        with tqdm(total=len(cls.image_paths)-1, file=sys.stdout) as pbar:
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                if not os.path.exists(output_filename):
                    try:
                        img_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        face_list, bbox = extract_faces(img_array, detector, args.detect_multiple_faces, args.threshold, image_size=160, margin=44)
                        if len(face_list) > 0:
                            for i, face in enumerate(face_list):
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                face_rgb = cv2.cvtColor(face_list[i], cv2.COLOR_BGR2RGB)
                                cv2.imwrite(output_filename_n,face_rgb)
                        else:
                            print('Unable to align "%s"' % image_path)
                pbar.update(1)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--threshold', type=float,
        help='Threshold for accepting the face, default is 0.8', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
