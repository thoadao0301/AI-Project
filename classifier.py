import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pickle
import cv2

from PIL import Image
from tqdm import tqdm
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Get embedded list images and list labels in folder
def load_dataset(dir, embedder):
    x, y, img_path = [], [] , []
    n = len(os.listdir(dir))
    i = 1
    for subdir in os.listdir(dir):
        print('Folder: ',i,'/',n)
        subdir_path = dir + '/' + subdir

        data_embedding,data_path = get_embedding(subdir_path, embedder)
        for j in range(len(data_embedding)):
            x.append(data_embedding[j])
            img_path.append(data_path[j])
            y.append(subdir)
        i+=1
    return x, y, img_path


def load_dataset_mtcnn(dir, embedder, detector, margin):
    x, y,img_path = [], [],[]
    n = len(os.listdir(dir))
    i = 1
    for subdir in os.listdir(dir):
        print('Folder: ',i,'/',n)
        subdir_path = dir + '/' + subdir

        data_embedding,data_path = get_embedding_mtcnn(subdir_path, embedder, detector,margin)
        for j in range(len(data_embedding)):
            x.append(data_embedding[j])
            img_path.append(data_path[j])
            y.append(subdir)
        i += 1
    return x, y, img_path

def get_embedding(dir, embedder):
    x,img_path = [],[]
    filename_list = os.listdir(dir)
    with tqdm(total=len(filename_list), file=sys.stdout) as pbar:
            for filename in filename_list:
                # set up file path
                filepath = os.path.join(dir,filename)
                # load image from file
                image = Image.open(filepath)
                img_path.append(filepath)
                # convert to RGB, if needed
                image = image.convert('RGB')
                #resize image
                image = image.resize((160,160))
                # convert to array
                pixels = tf.cast(np.array(image),tf.float32)/255
                # get embeddings for the faces in an image
                pixels = tf.expand_dims(pixels,axis=0)

                embedding = embedder.predict(pixels)[0]
                x.append(embedding)
                pbar.update(1)
    return x,img_path

def get_embedding_mtcnn(dir, embedder, detector, margin):
    x,img_path= [],[]
    filename_list = os.listdir(dir)
    with tqdm(total=len(filename_list), file=sys.stdout) as pbar:
            for filename in filename_list:
                # set up file path
                filepath = os.path.join(dir,filename)
                img_path.append(filepath)
                # load image from file
                image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
                # get image size
                img_size = np.asarray(image.shape)[0:2]
                # extract face, using default weights
                results = detector.detect_faces(image)
                if len(results)==0:
                    continue
                # extract the bounding box from the first face
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height

                x1 = np.maximum(int(x1- margin/2), 0)
                y1 = np.maximum(int(y1 - margin/2), 0)
                x2 = np.minimum(int(x2 + margin/2), int(img_size[1]))
                y2 = np.minimum(int(y2 + margin/2), int(img_size[0]))

                # extract the face
                face = image[y1:y2, x1:x2]
                # resize image
                image = Image.fromarray(face)
                image = image.resize((160,160))
                # convert to array
                image = tf.cast(np.array(image),tf.float32)/255
                # get embeddings for the faces in an image
                image = tf.expand_dims(image,axis=0)

                embedding = embedder.predict(image)[0]
                x.append(embedding)
                pbar.update(1)
    return x,img_path

def show_wrong_predict(img_paths,imgs_emd,labels,out_encoder,model):
    y_preds = model.predict(imgs_emd)
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
    num_classes = model.layers[-1].units
    model.load_weights(args.model_weights_path)
    embedder = tf.keras.models.Model(inputs=[model.input], outputs=[model.layers[-2].output])

    # # Create train dataset
    # if args.use_mtcnn_model:
    #     detector = MTCNN()
    #     train_x, train_y,_ = load_dataset_mtcnn(args.train_data_path, embedder, detector,args.marign)
    # else:
    #     train_x, train_y,_ = load_dataset(args.train_data_path, embedder)
    # # Create test dataset
    # if args.test_data_path is not None:
    #     if args.use_mtcnn_model:
    #         test_x, test_y,img_paths = load_dataset_mtcnn(args.test_data_path, embedder,detector, args.marign)
    #     else:
    #         test_x, test_y,img_paths = load_dataset(args.test_data_path, embedder)

    # For testing
    with open('test.npy','rb') as f:
        train_x=np.load(f)
        train_y=np.load(f)
        img_paths=np.load(f)
        test_x=np.load(f)
        test_y=np.load(f)
        # np.save(f,train_x)
        # np.save(f, train_y)
        # np.save(f, img_paths)
        # np.save(f, test_x)
        # np.save(f, test_y)





    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)

    # get class names:
    class_encode = set(train_y)
    class_names = [None] * len(class_encode)
    for i in range(len(class_names)):
        class_names[i] = out_encoder.inverse_transform([i])[0]

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

    # Create embedding model
    if args.mode == 'LOGISTIC':
        print('Training with Logistic model.')
        best_score = 0
        best_model = None
        max_iter = 10000
        if args.max_iter > 0:
            max_iter = args.max_iter
        for C in np.arange(args.C_min, args.C, args.C_step):
            print('Training with the Regularization parameter:', C)
            model = LogisticRegression(multi_class='multinomial', C=C, max_iter=max_iter)
            model.fit(train_x, train_y)
            # score
            score_train = model.score(train_x, train_y)
            print('Accuracy: train=%3f' % score_train)
            if args.test_data_path is not None:
                score_test = model.score(test_x, test_y)
                if score_test > best_score:
                    best_score = score_test
                    best_model = model
                print('Accuracy: test=%.3f' % score_test)
            else:
                if score_train >best_score:
                    best_model = model
                    best_score = score_train
            # Saving model
            if args.model_classifier_path is None:
                model_classifier = os.path.join(os.path.split(args.model_path)[0], 'model_classifier_Logistic.sav')
            else:
                model_classifier = args.model_classifier_path
            print('------------------------------------------')
            # save model
        with open(model_classifier, 'wb') as f:
            pickle.dump(best_model, f)
            pickle.dump(class_names, f)
        if args.show_wrong_predict and args.test_data_path is not None:
            show_wrong_predict(img_paths,test_x,test_y,out_encoder,best_model)

    elif args.mode=='SVM':
        print('Training with SVM model')
        best_score=0
        best_model=None
        for C in np.arange(args.C_min,args.C,args.C_step):
            print('Training with the Regularization parameter:', C)
            model = SVC(kernel='linear', probability=True)
            model.fit(train_x, train_y)
            # score
            score_train = model.score(train_x,train_y)
            print('Accuracy: train=%3f' % score_train)
            if args.test_data_path is not None:
                score_test = model.score(test_x,test_y)
                if score_test>best_score:
                    best_score=score_test
                    best_model=model
                print('Accuracy: test=%.3f' % score_test)
            else:
                if score_train >best_score:
                    best_model = model
                    best_score = score_train
            # Saving model
            if args.model_classifier_path is None:
                model_classifier=os.path.join(os.path.split(args.model_path)[0],'model_classifier_SVM.sav')
            else:
                model_classifier=args.model_classifier_path
            print('------------------------------------------')
            # save model
        with open(model_classifier,'wb') as f:
            pickle.dump(best_model, f)
            pickle.dump(class_names,f)
        if args.show_wrong_predict and args.test_data_path is not None:
            show_wrong_predict(img_paths,test_x,test_y,out_encoder,best_model)
    else:
        print('Training with KNN:')
        model = KNeighborsClassifier(n_neighbors=num_classes,algorithm='ball_tree',weights='distance')
        model.fit(train_x, train_y)
        # score
        score_train = model.score(train_x, train_y)
        print('Accuracy: train=%3f' % score_train)
        if args.test_data_path is not None:
            score_test = model.score(test_x, test_y)
            print('Accuracy: test=%.3f' % score_test)
        # Saving model
        if args.model_classifier_path is None:
            model_classifier = os.path.join(os.path.split(args.model_path)[0], 'model_classifier_KNN.sav')
        else:
            model_classifier = args.model_classifier_path
        # save model
        with open(model_classifier, 'wb') as f:
            pickle.dump(model, f)
            pickle.dump(class_names, f)
        if args.show_wrong_predict and args.test_data_path is not None:
            show_wrong_predict(img_paths,test_x,test_y,out_encoder,model)
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['KNN', 'SVM', 'LOGISTIC'],
                        help='Method for classification ', default='KNN')

    parser.add_argument('model_path', type=str,
                        help='Path to the model  training by softmax (.h5 file)')

    parser.add_argument('model_weights_path', type=str,
                        help='Path to the weights of model keras (.h5 file)')

    parser.add_argument('train_data_path', type=str,
                        help='Path to the train dataset directory')

    parser.add_argument('--model_classifier_path', type=str,
                        help='Path to save the classifier model, if training with softmax, the classifier model path is the model_path', default=None)

    parser.add_argument('--test_data_path', type=str,
                        help='Path to the test dataset directory', default=None)

    parser.add_argument('--use_mtcnn_model', type=bool,
                        help='True if the data input is raw data', default=False)

    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)

    parser.add_argument('--C', type=float,
                        help='Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. '
                             'The penalty is a squared l2 penalty. Use for classifier model', default=1.0)
    parser.add_argument('--C_min', type=float,
                        help='Use for C option. The start point of C', default=0.1)
    parser.add_argument('--C_step', type=float,
                        help='Use for C option. Step increase from C_min to C', default=0.1)

    parser.add_argument('--max_iter', type=int,
                        help='Maximum number of iterations taken for the solvers to converge.', default=-1)

    parser.add_argument('--show_wrong_predict', type=bool,
                        help='Show wrong predict images', default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))