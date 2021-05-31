import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
import random
from tqdm import tqdm

tf.config.run_functions_eagerly(True)


class TensorLog:

    def __init__(self):
        self.logFile = 'tensor_report_{}.csv'.format(datetime.datetime.now())
        open(self.logFile, 'w+').close()
        print('Logger initialised with {}'.format(self.logFile))

    @staticmethod
    def log(epoch, generator_loss, discriminator_loss):
        logToWrite = "{},{},{}\n".format(epoch, generator_loss, discriminator_loss)
        return logToWrite

    def writeLogs(self, epoch, generator_loss, discriminator_loss):
        with open(self.logFile, 'a') as logUtil:
            logUtil.write(TensorLog.log(epoch, generator_loss, discriminator_loss))


class Helpers:
    @staticmethod
    def random_hue(img):
        return tf.image.random_hue(img, random.randint(0, 1) * 0.5)

    @staticmethod
    def random_saturation(img):
        return tf.image.random_saturation(img, 5, 10)

    @staticmethod
    def random_brightness(img):
        return tf.image.random_brightness(img, 0.2)

    @staticmethod
    def random_contrast(img):
        return tf.image.random_contrast(img, 0.2, 0.5)


class Dataset:
    def __init__(self, x_path, y_path, img_size, resize_required, load, batch_size=128):
        self.x_path = x_path
        self.y_path = y_path
        self.x = np.array([])
        self.y = np.array([])
        self.dataset = None
        self.batch_size = batch_size
        self.resize_required = resize_required
        self.IMG_SIZE = img_size
        self.load = load
        self.make_synthetic_dataset()

    def make_synthetic_dataset(self):
        X = []
        Y = []
        if self.load:
            self.x = np.load(self.x_path)
            self.y = np.load(self.y_path)
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
            self.dataset = self.dataset.batch(self.batch_size)
        else:
            for img_name in tqdm(os.listdir(self.x_path)):
                try:
                    img = cv2.imread(os.path.join(self.x_path, img_name))
                    if self.resize_required:
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img = np.array(img) / 255
                    X.append(img)
                    img = cv2.imread(os.path.join(self.y_path, img_name))
                    if self.resize_required:
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img = np.array(img) / 255
                    Y.append(img)
                except:
                    pass
            X = tf.convert_to_tensor(np.array(X))
            Y = tf.convert_to_tensor(np.array(Y))
            self.dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            self.dataset = self.dataset.batch(self.batch_size)

    def augment(self, output_path):
        for img in os.listdir(self.x_path):
            try:
                base_img = tf.convert_to_tensor(cv2.imread(os.path.join(self.x_path, img)))
                augmentation_functions = [tf.image.flip_up_down, tf.image.flip_left_right,
                                          Helpers.random_contrast, Helpers.random_hue, Helpers.random_brightness,
                                          Helpers.random_saturation]
                for i in range(len(augmentation_functions)):
                    aug_img = augmentation_functions[i](base_img)
                    cv2.imwrite('{}'.format(os.path.join(output_path, '{}_{}'.format(i, img))), np.float32(aug_img))
            except Exception as e:
                print(e)


class GAN:
    def __init__(self, dataset=None, epochs=20):
        self.img_size = 100
        self.generator = tf.keras.models.Model()
        self.discriminator = tf.keras.models.Model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0005)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)
        self.epochs = epochs
        self.initialize_model()
        if dataset != None:
            self.dataset = dataset
            self.logger = TensorLog()
            self.fit()

    def get_generator_model(self):
        inputs = tf.keras.layers.Input(shape=(100, 100, 3))
        conv1 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1)(inputs)
        conv1 = tf.keras.layers.LeakyReLU()(conv1)
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
        conv1 = tf.keras.layers.LeakyReLU()(conv1)
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
        conv1 = tf.keras.layers.LeakyReLU()(conv1)

        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=1)(conv1)
        conv2 = tf.keras.layers.LeakyReLU()(conv2)
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
        conv2 = tf.keras.layers.LeakyReLU()(conv2)
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
        conv2 = tf.keras.layers.LeakyReLU()(conv2)

        conv3 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1)(conv2)
        conv3 = tf.keras.layers.LeakyReLU()(conv3)
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
        conv3 = tf.keras.layers.LeakyReLU()(conv3)
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
        conv3 = tf.keras.layers.LeakyReLU()(conv3)

        bottleneck = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(
            conv3)

        concat_1 = tf.keras.layers.Concatenate()([bottleneck, conv3])
        conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(concat_1)
        conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_3)
        conv_up_3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_3)

        concat_2 = tf.keras.layers.Concatenate()([conv_up_3, conv2])
        conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(concat_2)
        conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_2)
        conv_up_2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_2)

        concat_3 = tf.keras.layers.Concatenate()([conv_up_2, conv1])
        conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(concat_3)
        conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_1)
        conv_up_1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_1)

        self.generator = tf.keras.models.Model(inputs, conv_up_1)

    def get_discriminator_model(self):
        layers = [
            tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu', input_shape=(100, 100, 3)),
            tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        self.discriminator = tf.keras.models.Sequential(layers)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(
            tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1),
            real_output)
        fake_loss = self.cross_entropy(
            tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1),
            fake_output)
        #         real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        #         fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output, real_y):
        real_y = tf.cast(real_y, 'float32')
        return self.mse(fake_output, real_y)

    def initialize_model(self):
        self.get_generator_model()
        self.get_discriminator_model()

    @tf.function
    def train_step(self, input_x, real_y, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(input_x, training=True)
            real_output = self.discriminator(real_y, training=True)
            generated_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(generated_images, real_y)
            disc_loss = self.discriminator_loss(real_output, generated_output)
            self.logger.writeLogs(epoch, tf.keras.backend.eval(gen_loss), tf.keras.backend.eval(disc_loss))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def fit(self):
        for epoch in range(self.epochs):
            for (x, y) in tqdm(self.dataset.dataset):
                self.train_step(x, y, epoch)
        self.generator.save('g_model')
        self.discriminator.save('d_model')

    def load_models(self):
        self.generator = tf.keras.models.load_model('g_model')
        self.discriminator = tf.keras.models.load_model('d_model')

    def generate(self, test_path, output_path):
        for img_name in os.listdir(test_path):
            print(img_name)
            img = np.expand_dims(np.array(cv2.imread(os.path.join(test_path, img_name))), axis=0)
            predicted_img = np.array(self.generator([img]))[0]
            cv2.imwrite(os.path.join(output_path, img_name), predicted_img)
            print(os.path.join(output_path, img_name))

    def generate_img(self, img_path, output_path):
        img = np.expand_dims(np.array(cv2.imread(img_path)), axis=0)
        predicted_img = np.array(self.generator([img]))[0]
        cv2.imwrite(os.path.join(output_path), predicted_img)

    def inference(self, images_path, save_path, ground_truth_path=None):
        if ground_truth_path is not None:
            for img_n in tqdm(os.listdir(images_path)):
                try:
                    img = cv2.imread(os.path.join(images_path, img_n))
                    img_ = cv2.resize(img, (100, 100))
                    img_ = np.array(img_) / 255
                    img = np.expand_dims(img_, 0)
                    output = np.array(self.generator.predict([np.array(img)])[0])
                    target = cv2.imread(os.path.join(ground_truth_path, img_n))
                    target = np.array(cv2.resize(target, (100, 100)))
                    res = np.concatenate((img_ * 255, output * 255, target), axis=1)
                    cv2.imwrite('{}/generated_{}.jpg'.format(save_path, os.path.splitext(img_n)[0]), res)
                    print('Gen')
                except Exception as e:
                    print(e)


dataset = Dataset(x_path='source', y_path='target', img_size=100, resize_required=True, load=False)
gan = GAN(dataset)
