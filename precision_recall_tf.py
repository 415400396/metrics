from tensorflow import keras
from numpy.linalg import norm
from numpy import subtract
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import os

parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the real images and generated images '
                          ))

def pretrained_model(img_shape, num_classes, layer_type):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(keras_input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation=layer_type, name='fc1')(x)
    x = Dense(4096, activation=layer_type, name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return pretrained_model

def get_npdata(name_imgs_train):
    X_train = []
    for i, myid in enumerate(name_imgs_train):
        image = load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
        image = img_to_array(image)/255.0
        X_train.append(image)
    X_train = np.array(X_train)
    return(X_train)

def pairwise_distances(U,V):
    row = U.shape[0]
    col = V.shape[0]
    pair_dist = np.zeros([row, col], dtype=np.float16)
    for i in range(row):
        for j in range(col):
            pair_dist[i,j] = norm(subtract(U[i,:], V[j,:]))
    return pair_dist

# estimate the manifold for each features
def manifold_estimator(a_features, b_features, nhood_size, row_batch_size = 1000, col_batch_size = 2000):
    """
        Args :
            a_features (np.array/tf.Tensor) : Feature vectors of phi_a
            b_features (np.array/tf.Tensor) : Feature vectors of phi_b
            nhood_size : neighborhood size k
        Returns :
            fraction value of b_features within the estimated manifold of a_features
    """
    num_images = a_features.shape[0]
    #Approximate manifold of phi_a
    D = np.zeros([num_images, 1])
    distance_batch = np.zeros([row_batch_size, num_images], dtype = np.float16)
    ab_distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float16)
    batch_prediction = np.zeros([num_images, 1], dtype = np.int32)
    #seq = np.arrange(nhood_size + 1, dtype = np.int32)


    for begin1 in range(0, num_images, row_batch_size):
        end1 = min(begin1 + row_batch_size, num_images)
        row_batch = a_features[begin1:end1]

        for begin2 in range(0, num_images, col_batch_size):
            end2 = min(begin2 + col_batch_size, num_images)
            col_batch = a_features[begin2:end2]
            distance_batch[0:end1-begin1, begin2:end2] = pairwise_distances(row_batch, col_batch)

        D[begin1:end1,0] = np.partition(distance_batch[0:end1-begin1, :], nhood_size + 1)[:,nhood_size]


    for begin3 in range(0, num_images, row_batch_size):
        end3 = min(begin3 + row_batch_size, num_images)
        feature_batch = b_features[begin3:end3]

        for begin4 in range(0, num_images, col_batch_size):
            end4 = min(begin4 + col_batch_size, num_images)
            ref_batch = a_features[begin4:end4]

            ab_distance_batch[0:end3-begin3, begin4:end4] = pairwise_distances(feature_batch, ref_batch)

        samples_in_manifold = ab_distance_batch[0:end3-begin3, :] <= D[0:end3-begin3]
        batch_prediction[begin3:end3,0] = np.any(samples_in_manifold, axis = 1).astype(np.int32)

    return sum(batch_prediction)/num_images

def knn_precision_recall(real_features, gen_features, k = 3):
    precision = manifold_estimator(real_features, gen_features, k, row_batch_size=10, col_batch_size= 20)
    recall = manifold_estimator(gen_features, real_features, k, row_batch_size=10, col_batch_size= 20 )

    print("Precision: {:>6.4f} Recall: {:>6.4f}".format(float(precision), float(recall)))
    return float(precision), float(recall)




def main():
    args = parser.parse_args()
    iamges_real = np.sort(os.listdir(args.path[0]))
    images_generated = np.sort(os.listdir(args.path[1]))
    X_real = get_npdata(iamges_real)
    X_gen = get_npdata(images_generated)
    model = pretrained_model(X_real.shape[1:], 10, 'relu')
    real_feature_vector = model.predict(X_real)
    gen_feature_vector = model.predict(X_gen)
    knn_precision_recall(real_feature_vector, gen_feature_vector)


if __name__ == '__main__':
    main()
