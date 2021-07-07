
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
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import cv2
import pickle

from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Get embedded list images and list labels in folder
def load_dataset(dir):
    x, y = [], []
    n = len(os.listdir(dir))
    i = 1
    for subdir in os.listdir(dir):
        print('Folder: ',i,'/',n)
        subdir_path = os.path.join(dir,subdir)

        data_path = get_image_paths(subdir_path)
        for data in data_path:
            x.append(data)
            y.append(subdir)
        i+=1
    return x, y


def get_image_paths(dir):
    x = []
    filename_list = os.listdir(dir)
    with tqdm(total=len(filename_list), file=sys.stdout) as pbar:
            for filename in filename_list:
                file_path = os.path.join(dir,filename)
                x.append(file_path)
                pbar.update(1)
    return x

def show_wrong_predict(img_paths,y_preds,labels,out_encoder):
    idx_diff = np.flatnonzero(np.array(y_preds) != np.array(labels))
    num_wrong_predict = len(idx_diff)
    print('Number of wrong predict in test set: ', num_wrong_predict)
    # display image and labels

    plt.figure(figsize=(15, 15))
    k=1
    for j in range(0,num_wrong_predict,25):
        for i in range(j, j+25):
            image = cv2.imread(img_paths[idx_diff[i]])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(5, 5, k)
            plt.imshow(image)
            text =  str(out_encoder.inverse_transform([labels[idx_diff[i]]])[0]) + '/' + str(out_encoder.inverse_transform([y_preds[idx_diff[i]]])[0])
            plt.title(text,fontsize=13)
            plt.axis("off")
            k+=1
        k=1
        plt.show()
def main(args):

    # Create facenet model with L2 embeddings, load weights from pretrained model
    model = tf.keras.models.load_model(args.model_path)
    model.load_weights(args.model_weights_path)

    # Create train dataset
    train_x, train_y = load_dataset(args.train_data_path)
    # Create test dataset
    if args.test_data_path is not None:
        test_x, test_y= load_dataset(args.test_data_path)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)

    # get class names:
    class_encode = set(train_y)
    class_names = [None] * len(class_encode)

    for i in range(len(class_names)):
        class_names[i] = out_encoder.inverse_transform([i])[0]
    with open(os.path.join(os.path.split(args.model_path)[0], 'class_name.sav'), 'wb') as f:
        pickle.dump(class_names, f)
    with open(os.path.join(os.path.split(args.model_path)[0], 'class_name.txt'), 'w') as f:
        for i in range(len(class_names)):
            text = str(i)+':'+str(class_names[i]) + '\n'
            f.write(text)
    if args.test_data_path is not None:
        test_y = out_encoder.transform(test_y)

    # tranning model
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print('Training images shape: ', train_x.shape)
    print('Training images labels: ',train_y.shape)
    if args.test_data_path is not None:
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        print('Testing images shape: ', test_x.shape)
        print('Testing images labels: ', test_y.shape)

    cal_acc = tf.keras.metrics.Accuracy()

    print('Evaluate on training set')
    y_preds_train=np.zeros_like(train_y)
    n = len(train_x)
    with tqdm(total=n-1, file=sys.stdout) as pbar:
        for i in range(n):
            file_path = train_x[i]
            image = Image.open(file_path)
            image = image.resize((160,160))
            image = tf.cast(np.array(image),tf.float32)/255
            image = tf.expand_dims(image, axis=0)
            predict = np.argmax(model.predict(image))
            y_preds_train[i]= predict
            pbar.update(1)

    cal_acc.update_state(y_preds_train, train_y)
    score_train = cal_acc.result().numpy()
    print('Accuracy: train=%3f' % score_train)

    if args.test_data_path is not None:
        cal_acc.reset_state()
        print('Evaluate on testing set')
        y_preds_test = np.zeros_like(test_y)
        n = len(test_x)
        with tqdm(total=n - 1, file=sys.stdout) as pbar:
            for i in range(n):
                file_path = test_x[i]
                image = Image.open(file_path)
                image = image.resize((160, 160))
                image = tf.cast(np.array(image), tf.float32) / 255
                image = tf.expand_dims(image, axis=0)
                predict = np.argmax(model.predict(image))
                y_preds_test[i] = predict
                pbar.update(1)
        cal_acc.update_state(y_preds_test, test_y)
        score_test = cal_acc.result().numpy()
        print('Accuracy: test=%3f' % score_test)
        if args.show_wrong_predict and args.test_data_path is not None:
            show_wrong_predict(test_x,y_preds_test,test_y,out_encoder)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model  training by softmax (.h5 file)')

    parser.add_argument('model_weights_path', type=str,
                        help='Path to the weights of model keras (.h5 file)')

    parser.add_argument('train_data_path', type=str,
                        help='Path to the train dataset directory')

    parser.add_argument('--test_data_path', type=str,
                        help='Path to the test dataset directory', default=None)

    parser.add_argument('--show_wrong_predict', type=bool,
                        help='Show wrong predict images', default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))