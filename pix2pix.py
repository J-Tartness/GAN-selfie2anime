from __future__ import print_function, division

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def load_data(path1, path2):
    imgs_A = []
    imgs_B = []
    filename1 = os.listdir(path1)
    filename2 = os.listdir(path2)

    for i in range(len(filename1)):
        imgA = np.array(cv2.imread(os.path.join(path1,filename1[1])))
        imgB = np.array(cv2.imread(os.path.join(path2,filename2[1])))
        
        imgA = imgA.astype("float32")
        imgB = imgB.astype("float32")
        
        if np.random.random() < 0.5:
            imgA = np.fliplr(imgA)
            imgB = np.fliplr(imgB)
                
        imgA = (imgA/ 127.5) -1
        imgB = (imgB/ 127.5) -1
        
        imgs_A.append(imgA)
        imgs_B.append(imgB)
        
    return np.array(imgs_A), np.array(imgs_B)

trainA, trainB = load_data('./testA', './testB')

# Input shape
img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)

# Calculate output shape of D (PatchGAN)
patch = int(img_rows / 2**4)
disc_patch = (patch, patch, 1)

# Number of filters in the first layer of G and D
gf = 64
df = 64

optimizer = Adam(0.0002, 0.5)

def build_generator():
    """U-Net Generator"""
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

def build_discriminator():
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

def sample_images(generator, epoch):
    os.makedirs('./output', exist_ok=True)
    r, c = 3, 3

    idx = [0,1,2]
    imgs_A, imgs_B = trainA[idx], trainB[idx]
    fake_A = generator.predict(imgs_B)

    gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./output/%d.png" % epoch)
    plt.close()

def train(generator, discriminator, combined, epochs, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    for epoch in range(epochs):
        idx = np.random.randint(0, trainA.shape[0], batch_size)
        imgs_A, imgs_B = trainA[idx], trainB[idx]

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Condition on B and generate a translated version
        fake_A = generator.predict(imgs_B)

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------

        # Train the generators
        g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                d_loss[0], 100*d_loss[1],
                                                                g_loss[0],
                                                                elapsed_time))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(generator,epoch)

if __name__ == "__main__":
    discriminator = build_discriminator()
    discriminator.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])

    #-------------------------
    # Construct Computational
    #   Graph of Generator
    #-------------------------

    # Build the generator
    generator = build_generator()

    # Input images and their conditioning images
    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)

    # By conditioning on B generate a fake version of A
    fake_A = generator(img_B)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = discriminator([fake_A, img_B])

    combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
    combined.compile(loss=['mse', 'mae'],
                          loss_weights=[1, 100],
                          optimizer=optimizer)
    
    train(generator, discriminator, combined, 200, 5, 10)