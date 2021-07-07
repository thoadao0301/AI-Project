# Finding people in video with MTCNN and Facenet
This is a project using the Facenet model described in the paper ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832) and MTCNN model described in the paper ["Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"](https://arxiv.org/abs/1604.02878) for finding people in video.
# Compatibility
```
python: 3.7
tensorflow : 2.5.0
numpy: 1.19.5
cv2: 4.5.2
mtcnn: 0.1.0
PIL: 8.2.0
matplotlib: 3.4.2
sklearn: 0.24.2
scipy: 1.1.0
```

# Pre-trained models
Model training with softmax: [FaceNet-softmax](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)

Model training with triplet loss: [Facenet-triplet](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz)
# Training data
The data is provided by [Mr. Vinh Truong Hoang](https://sites.google.com/view/vinhsiam)

If you want to access the dataset, please send an email to Mr. Hoang for permission.

# Pre-processing
## Data Preprocessing
In our project, we use the model mtcnn has been implemented using Keras for extracting face in images. You can find the model in [here](https://pypi.org/project/mtcnn/). We also provide a method for dividing datasets and creating more data for training from raw data. The detail of Data preprocessing [here](https://github.com/thoadao0301/AI-Project/wiki/Data-Preprocessing)

# Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/thoadao0301/AI-Project/wiki/Classifier-training-of-Inception-ResNet-v1-using-Softmax).

The implement training using triplet loss, we using the code which is implemented by [David Sandberg](https://github.com/davidsandberg). The project can be found in [here](https://github.com/davidsandberg/facenet).


