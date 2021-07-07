import tensorflow_addons as tfa
import tensorflow as tf
import os
import numpy as np

model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(160,160,3))
avgpool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
dropout = tf.keras.layers.Dropout(0.3)(avgpool)
dense = tf.keras.layers.Dense(units=128)(dropout)
L2_normalization = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis = 1), name='L2_normalization')(dense)
model = tf.keras.models.Model(inputs = [model.input], outputs = [L2_normalization])

def scheduler(epoch,lr):
    if epoch < 50:
        return 0.1
    elif epoch < 100:
        return 0.01
    elif epoch < 150:
        return 0.001
    else:
        return 0.0001

BATCH_SIZE = 16
# Create data train and validation from directory
datagen_train = tf.keras.preprocessing.image_dataset_from_directory(r'E:\Study\AIP\model_male\dataset\mtcnn_dataset_182_male_full',
                                                                    batch_size=BATCH_SIZE,seed=123,
                                                            labels='inferred', label_mode="int",image_size=(160,160),
                                                            validation_split=0.2,subset="training")
datagen_val   = tf.keras.preprocessing.image_dataset_from_directory(r'E:\Study\AIP\model_male\dataset\mtcnn_dataset_182_male_full',
                                                                    batch_size=BATCH_SIZE,seed=123,
                                                            labels='inferred', label_mode="int", image_size=(160, 160),
                                                            validation_split=0.2, subset="validation")

# Create facenet model with L2 embeddings
out_model=os.path.join(r'E:\Study\AIP\model_male\test','triplet_keras_facenet_model.h5')
model.save(out_model)

# Rescaling dataset
AUTOTUNE = tf.data.AUTOTUNE

train_ds = datagen_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = datagen_val.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_vds= val_ds.map(lambda x, y: (normalization_layer(x), y))

# Create checkpoint
MODEL_CHECKPOINT_PATH=os.path.join(r'E:\Study\AIP\model_male\test','model_triplet_checkpoint.h5')
mcp = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor="val_accuracy", save_weights_only=True)

lr_decrease = tf.keras.callbacks.LearningRateScheduler(scheduler)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=r'E:\Study\AIP\model_male\test\logs_test', histogram_freq=1)
callbacks=[mcp,lr_decrease,tensorboard_callback]

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss=tfa.losses.TripletHardLoss())

# Training
model.fit(train_ds,validation_data=val_ds,validation_freq=1, epochs=200,callbacks=callbacks,batch_size=BATCH_SIZE)