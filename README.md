# Finding people in video with MTCNN and Facenet
This is a project using the Facenet model described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering" and MTCNN model described in the paper Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks for finding people in video.
# Compatibility
python: 3.7
tensorflow : 2.5.0
numpy: 1.19.5
cv2: 4.5.2
mtcnn: 0.1.0
PIL: 8.2.0
matplotlib: 3.4.2
sklearn: 0.24.2
scipy: 1.1.0

# Pre-trained models
Model training with softmax: https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
Model training with triplet loss: https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz
# Training data
The data is provided by Mr. Vinh Truong Hoang.
https://sites.google.com/view/vinhsiam

If you want to access the dataset, please send an email to Mr. Hoang for permission.

# Pre-processing
## Face alignment using MTCNN
data_preprocessing
python align_dataset_mtcnn.py input_dir output_dir --image_size 160 --margin 44 --detect_multiple_faces --threshold
python create_test_dataset.py input_dir output_dir --size
python data_generator.py data_dir

training
python softmax_training.py model_path dataset_path --model_checkpoint_path --model_logs_path --learning_rate --epochs --batch_size --seed_random --validation_split -- validation_freq

classifier
python classifier_softmax_facenet.py model_path model_weights_path train_data_path --test_data_path --use_mtcnn_model --threshold_mtcnn_model --margin --show_wrong_predict
python classifier_triplet_facenet.py mode model_path model_weights_path train_data_path --test_data_path --use_mtcnn_model --threshold_mtcnn_model --margin --show_wrong_predict
output
python output_function.py model_facenet_path model_weights_path model_classfier_path input_video output_loc --export_video --frame_skip --threshold --id
