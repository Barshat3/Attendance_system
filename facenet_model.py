import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model

def inception_resnet_block(x, scale, block_type, activation='relu'):
    if block_type == 'block35':
        branch_0 = Conv2D(32, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(32, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(32, 3, activation=activation, padding='same')(branch_1)
        branch_2 = Conv2D(32, 1, activation=activation, padding='same')(x)
        branch_2 = Conv2D(32, 3, activation=activation, padding='same')(branch_2)
        branch_2 = Conv2D(32, 3, activation=activation, padding='same')(branch_2)
        mixed = tf.keras.layers.Concatenate()([branch_0, branch_1, branch_2])
        up = Conv2D(x.shape[-1], 1, activation=None, padding='same')(mixed)
        x = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, output_shape=x.shape[1:])([x, up])
        if activation is not None:
            x = Activation(activation)(x)

    elif block_type == 'block17':
        branch_0 = Conv2D(128, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(128, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(128, [1, 7], activation=activation, padding='same')(branch_1)
        branch_1 = Conv2D(128, [7, 1], activation=activation, padding='same')(branch_1)
        mixed = tf.keras.layers.Concatenate()([branch_0, branch_1])
        up = Conv2D(x.shape[-1], 1, activation=None, padding='same')(mixed)
        x = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, output_shape=x.shape[1:])([x, up])
        if activation is not None:
            x = Activation(activation)(x)

    elif block_type == 'block8':
        branch_0 = Conv2D(192, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(192, 1, activation=activation, padding='same')(x)
        branch_1 = Conv2D(192, [1, 3], activation=activation, padding='same')(branch_1)
        branch_1 = Conv2D(192, [3, 1], activation=activation, padding='same')(branch_1)
        mixed = tf.keras.layers.Concatenate()([branch_0, branch_1])
        up = Conv2D(x.shape[-1], 1, activation=None, padding='same')(mixed)
        x = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, output_shape=x.shape[1:])([x, up])
        if activation is not None:
            x = Activation(activation)(x)

    return x

def FaceNet_InceptionResNet(input_shape=(160, 160, 3), embedding_dim=128):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Initial Pooling Layer
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception Block)
    x = inception_resnet_block(x, scale=0.17, block_type='block35')

    # Inception-ResNet Blocks/home/barshat/Desktop/Attendence System/dataset/lfw-deepfunneled
    for _ in range(10):
        x = inception_resnet_block(x, scale=0.17, block_type='block35')

    # Reduction A
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(20):
        x = inception_resnet_block(x, scale=0.1, block_type='block17')

    # Reduction B
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8')

    # Final Pooling and Dense Layer for Embedding
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim)(x)
    outputs = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    model = Model(inputs, outputs, name="FaceNet_InceptionResNet")
    return model

# Instantiate the FaceNet model
facenet_model = FaceNet_InceptionResNet()
facenet_model.summary()

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss