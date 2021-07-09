"""Using model in finding people in video."""
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
import numpy as np
import cv2
import pickle
import argparse
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from mtcnn import MTCNN
from PIL import Image


def extract_faces(img_array, detector, image_size=160, margin=44):
    faces_list, bbox = [], []
    # convert channel
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)
    # extract the bounding box from the first face
    for face in results:
        confidence = face['confidence']
        if confidence < 0.8:
            continue
        face_bbox = face['box']
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
        face = face.resize((image_size,image_size))
        faces_list.append(face)
        bbox.append(face_bbox)

    return faces_list, bbox

def draw_bbox(image,bbox,text):
  x,y,w,h = bbox
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(image, text,(x, y),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  return image

def detect_face(image, id, model ,detector, threshold=0.5):
    # Get model and class_names
    model, class_names = model
    # create the detector, using default weights
    faces_list, bboxs_list = extract_faces(image, detector, image_size=160)
    # loop through each face in detections
    for i in range(len(faces_list)):

        # get embedded image
        # Get input and output tensors
        img = faces_list[i].convert('RGB')
        # resize image
        img = tf.cast(np.array(img), tf.float32) / 255
        # get embeddings for the faces in an image
        img = tf.expand_dims(img, axis=0)

        predictions = model.predict(img)

        # get coordinates (x,y) and weight, height (w, h) of the bounding box
        bbox = bboxs_list[i]

        # get predict from model
        class_index = np.argmax(predictions, axis=1)

        # get label name
        predict_name = class_names[class_index[0]]

        # get probability
        class_probability = predictions[0, class_index]
        if class_probability > threshold and class_index == id:
            text = f'{predict_name}:{class_probability}'
            image = draw_bbox(image, bbox, text)
    return image

def main(args):

    with tf.device('/GPU:0'):
        # Load facenet model
        print('Loading feature extraction model')
        model = tf.keras.models.load_model(args.model_path)
        model.load_weights(args.model_weights_path)
        # Create mtcnn model
        detector = MTCNN()
        # Load class name
        with open(args.class_names, 'rb') as infile:
            class_names = pickle.load(infile)

        image = cv2.imread(args.image_path)

        image = detect_face(image, args.id, (model, class_names), detector, args.threshold)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the facenet model training with softmax')
    parser.add_argument('model_weights_path', type=str,
                        help='Path to the facenet model weights training with softmax')
    parser.add_argument('class_names', type=str,
                        help='Path to the class_names in training classifier model')
    parser.add_argument('image_path', type=str,
                        help='Path to the image')
    parser.add_argument('--threshold', type=float,
                        help='Threshold for predict image', default=0.5)
    parser.add_argument('--id', type=int,
                        help='ID of specific person', default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))