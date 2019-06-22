from tensorflow.keras import layers as ly
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from CelebA_batchGenerator import CelebA

import os
import sys
import datetime
import numpy as np
from cv2 import imwrite, resize, INTER_AREA
from shutil import copyfile

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# dpo - Dropout.
# Bn - Batchnormalization.
# Mxp - Maxpooling.
# CnvTs - Conv2DTranspose s is k-size.
# dCnvs - Double Convolution layer.
# RFb - Real/Fake individual batch.
# ACly - Original ACGAN layers.
# '-' - Modify from.
# '+' - Add to.
# rdF - Reduce Filter.

TRAIN_NAME = 'ACGAN(Adam0002 ImgL RFb ACly-G3CnvTs3rdF-DMxp-D3dCnvs5)'
SAVE_FOLDER = 'Face_ACGAN_Saved/' + TRAIN_NAME


class ACGAN:
    def __init__(self, input_shape, num_class):
        # Input shape
        self.input_shape = input_shape
        self.num_classes = num_class
        self.latent_dim = 100
        self.d_loss_list = []
        self.g_loss_list = []
        self.p_real_list = []
        self.dis_acc_list = []
        self.cls_acc_list = []

        g_optimizer = Adam(0.0002, 0.5)
        d_optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = self.build_combinator()
        self.combined.compile(loss=losses,
                              optimizer=g_optimizer)

        self.discriminator.trainable = True
        self.discriminator.compile(loss=losses,
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

    def generate_noise(self, n):
        noises = np.random.normal(0, 1, (n, self.latent_dim))
        labels = np.random.randint(0, self.num_classes, (n, 1))
        return noises, labels

    def build_combinator(self):
        noise = ly.Input(shape=(self.latent_dim,))
        label = ly.Input(shape=(1,))
        img = self.generator([noise, label])

        self.discriminator.trainable = False
        valid, target_label = self.discriminator(img)

        return Model([noise, label], [valid, target_label])

    def build_generator(self):
        model = Sequential()
        cks = 3
        # Dense
        model.add(ly.Dense(64 * 10 * 8, activation="relu", input_dim=self.latent_dim))
        # Reshape to Image
        model.add(ly.Reshape((10, 8, 64)))
        # Reverse Conv_1
        model.add(ly.Conv2DTranspose(64, kernel_size=cks, strides=2, padding='same'))
        model.add(ly.Activation("relu"))
        # Reverse Conv_2
        model.add(ly.Conv2DTranspose(32, kernel_size=cks, strides=2, padding='same'))
        model.add(ly.Activation("relu"))
        # Reverse Conv_3
        model.add(ly.Conv2DTranspose(16, kernel_size=cks, strides=2, padding='same'))
        model.add(ly.Activation("relu"))
        # Conv_3
        model.add(ly.Conv2D(self.input_shape[-1], kernel_size=cks, padding='same'))
        model.add(ly.Activation("tanh"))

        model.summary()
        plot_model(model, os.path.join(SAVE_FOLDER, 'generator_model.png'),
                   show_shapes=True)

        noise = ly.Input(shape=(self.latent_dim,))
        label = ly.Input(shape=(1,), dtype='int32')
        label_embedding = ly.Flatten()(ly.Embedding(self.num_classes,
                                                    self.latent_dim)(label))

        model_input = ly.multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        cks = 5
        # Conv_1
        model.add(ly.Conv2D(16, kernel_size=cks, strides=1, padding="same",
                            input_shape=self.input_shape))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.Conv2D(16, kernel_size=cks, strides=1, padding='same'))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.MaxPooling2D(pool_size=2))
        # Conv_2
        model.add(ly.Conv2D(32, kernel_size=cks, strides=1, padding="same"))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.Conv2D(32, kernel_size=cks, strides=1, padding='same'))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.MaxPooling2D(pool_size=2))
        # Conv_3
        model.add(ly.Conv2D(64, kernel_size=cks, strides=1, padding="same"))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.Conv2D(64, kernel_size=cks, strides=1, padding='same'))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.MaxPooling2D(pool_size=2))
        """# Conv_4
        model.add(ly.Conv2D(128, kernel_size=cks, strides=1, padding="same"))
        model.add(ly.LeakyReLU(alpha=0.2))
        model.add(ly.Conv2D(128, kernel_size=cks, strides=1, padding="same"))
        model.add(ly.LeakyReLU(alpha=0.2))
        """
        # Flat.
        model.add(ly.Flatten())
        model.summary()

        img = ly.Input(shape=self.input_shape)
        plot_model(model, os.path.join(SAVE_FOLDER, 'discriminator_model.png'),
                   show_shapes=True)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = ly.Dense(1, activation="sigmoid")(features)
        label = ly.Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, batch_gen, epochs, sample_interval=50):
        total_time = datetime.timedelta()  # Time consume.

        for epoch in range(epochs):
            st = datetime.datetime.now()

            print('\nEpoch : %d/%d' % (epoch, epochs))
            d_loss_epoch = []
            g_loss_epoch = []
            dis_acc_epoch = []
            cls_acc_epoch = []

            batch_gen.shuffle_data()
            for i in range(len(batch_gen)):
                #  Train Discriminator.
                # ---------------------
                # Select next batch.
                # Labels with Gender 0 or 1 (male/female).
                imgs, imgs_labels = batch_gen.get_batch_gender(i)

                # Sample noise and label as generator input.
                noise, sampled_labels = self.generate_noise(batch_gen.batch_size)

                # Generate a half batch of new images.
                gen_imgs = self.generator.predict([noise, sampled_labels])

                # Adversarial ground truths
                valid = np.ones((batch_gen.batch_size, 1))
                fake = np.zeros((batch_gen.batch_size, 1))

                # Train the discriminator
                # Individual real-fake batch.
                d_loss_real = self.discriminator.train_on_batch(imgs, [valid, imgs_labels])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                d_loss_epoch.append(d_loss[0])
                dis_acc_epoch.append(d_loss[3])
                cls_acc_epoch.append(d_loss[4])

                #  Train Generator
                # ---------------------
                # Train the generator
                noise, sampled_labels = self.generate_noise(batch_gen.batch_size)

                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
                self.discriminator.trainable = True
                g_loss_epoch.append(g_loss[0])

                # Resulting.
                # ----------
                # Print the progress.
                progress = '%d/%d [D loss: %f, dis_acc.: %.2f%%, cls_acc.: %.2f%%] [G loss: %f]' \
                           % (i, len(batch_gen), d_loss[0], 100 * d_loss[3],
                              100 * d_loss[4], g_loss[0])
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()

            et = datetime.datetime.now() - st  # Count time only on train process.
            total_time = total_time + et

            # Test Generator.
            pred_real = self.test_generator(epochs)
            print('\nG fool D: %f' % pred_real)

            # Save weights.
            self.discriminator.save_weights(os.path.join(SAVE_FOLDER, 'discriminator.h5'))
            self.generator.save_weights(os.path.join(SAVE_FOLDER, 'generator.h5'))

            # Plot progress.
            self.d_loss_list.append(np.mean(d_loss_epoch))
            self.g_loss_list.append(np.mean(g_loss_epoch))
            self.dis_acc_list.append(np.mean(dis_acc_epoch))
            self.cls_acc_list.append(np.mean(cls_acc_epoch))
            self._plot_loss(epochs)
            self._plot_acc(epochs)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_sample(epoch)

        # Summery Time.
        time_file = os.path.join(SAVE_FOLDER, 'time_consume.txt')
        txt = 'Generator param : ' + str(self.generator.count_params())
        txt = txt + '\nDiscriminator param : ' + str(self.discriminator.count_params())
        txt = txt + '\nTotal Epochs : ' + str(epochs)
        txt = txt + '\nTotal time : ' + str(total_time)
        with open(time_file, 'w') as fil:
            fil.write(txt)

    def test_generator(self, max_epoch):
        noise, _ = self.generate_noise(1000)
        labels = np.zeros(1000, dtype=np.int32)
        labels[500:] = 1
        gen_imgs = self.generator.predict([noise, labels])

        dis_pred = self.discriminator.predict(gen_imgs)[0]
        pred_real = np.mean(dis_pred)

        self.p_real_list.append(pred_real)
        plt.figure()
        plt.plot(self.p_real_list, color='r')

        plt.legend('G confident')
        plt.xlim([0, max_epoch])
        plt.xlabel('Epochs')
        plt.ylabel('% confident (avg 1000 imgs)')
        plt.title(TRAIN_NAME + ' G Confident')

        plt.savefig(os.path.join(SAVE_FOLDER, 'G-confident.png'))
        plt.close()

        return pred_real

    def save_sample(self, epoch, save_name=None):
        r, c = 6, 6
        noise, _ = self.generate_noise(r * c)
        sampled_labels = np.zeros(r * c, dtype=np.int32)
        sampled_labels[int((r * c) / 2):] = 1
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1

        if save_name is None:
            path = os.path.join(SAVE_FOLDER, 'epochs_sample')
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, 'epoch-%d.png' % epoch))
        else:
            fig.savefig(os.path.join(SAVE_FOLDER, save_name))

        plt.close()

    def gen_an_image(self, label, noise=None, save_name=None, size=None):
        if noise is None:
            noise, _ = self.generate_noise(1)
        img = self.generator.predict([noise, np.array([label])])[0]
        img = ((0.5 * img + 0.5) * 255).astype('uint8')

        if size is not None:
            img = resize(img, size, INTER_AREA)

        if save_name is not None:
            imwrite(os.path.join(SAVE_FOLDER, save_name), img[:, :, ::-1])

        return img

    def _plot_loss(self, max_epoch):
        plt.figure()
        plt.plot(self.d_loss_list, color='b')
        plt.plot(self.g_loss_list, color='g')

        plt.legend(['Discriminator', 'Generator'])
        plt.xlim([0, max_epoch])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(TRAIN_NAME + ' Loss')

        plt.savefig(os.path.join(SAVE_FOLDER, 'loss-graph.png'))
        plt.close()

    def _plot_acc(self, max_epoch):
        plt.figure()
        plt.plot(self.dis_acc_list, color='c')
        plt.plot(self.cls_acc_list, color='m')

        plt.legend(['discriminate_acc', 'class_acc'])
        plt.xlim([0, max_epoch])
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.title(TRAIN_NAME + ' discriminator acc')

        plt.savefig(os.path.join(SAVE_FOLDER, 'dis_acc-graph.png'))
        plt.close()


if __name__ == '__main__':
    # Load Data.
    gen = CelebA(64, 80, batch_size=128, expand=(15, 43, 22, 15))

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    copyfile(os.path.basename(__file__),
             os.path.join(SAVE_FOLDER, os.path.basename(__file__)))

    gan = ACGAN((80, 64, 3), 2)

    # Load weights.
    #WEIGHT_FOLDER = 'Face_ACGAN_Saved/ACGAN(Adadelta)'
    #gan.generator.load_weights(WEIGHT_FOLDER + '/generator.h5')
    #gan.discriminator.load_weights(WEIGHT_FOLDER + '/discriminator.h5')

    # Save model yaml.
    with open(SAVE_FOLDER + '/generator.yaml', 'w') as yf:
        yf.write(gan.generator.to_yaml())
    with open(SAVE_FOLDER + '/discriminator.yaml', 'w') as yf:
        yf.write(gan.discriminator.to_yaml())

    gan.train(gen, epochs=50, sample_interval=5)
    gan.save_sample(1, 'final_result.png')