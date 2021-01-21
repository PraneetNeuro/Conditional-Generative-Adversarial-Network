import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm


class Dataset:
    def __init__(self, x_path, y_path, img_size, resize_required, load, batch_size=64):
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
                img = cv2.imread(os.path.join(self.x_path, img_name))
                if self.resize_required:
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                img = np.array(img) / 255
                X.append(img)
            for img_name in tqdm(os.listdir(self.y_path)):
                img = cv2.imread(os.path.join(self.y_path, img_name))
                if self.resize_required:
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                img = np.array(img) / 255
                Y.append(img)
            X = tf.convert_to_tensor(np.array(X))
            Y = tf.convert_to_tensor(np.array(Y))
            self.dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            self.dataset = self.dataset.batch(self.batch_size)


class GAN:
    def __init__(self, dataset, epochs=20):
        self.img_size = dataset.IMG_SIZE
        self.dataset = dataset
        self.generator = tf.keras.models.Model()
        self.discriminator = tf.keras.models.Model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0005)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)
        self.epochs = epochs
        self.initialize_model()
        self.fit()

    def get_generator_model(self):
        inputs = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
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

        bottleneck = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(conv3)

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
        # real_loss = self.cross_entropy(
        #     tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1),
        #     real_output)
        # fake_loss = self.cross_entropy(
        #     tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1),
        #     fake_output)
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output, real_y):
        real_y = tf.cast(real_y, 'float32')
        return self.mse(fake_output, real_y)

    def initialize_model(self):
        self.get_generator_model()
        self.get_discriminator_model()

    @tf.function
    def train_step(self, input_x, real_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(input_x, training=True)
            real_output = self.discriminator(real_y, training=True)
            generated_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(generated_images, real_y)
            disc_loss = self.discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def fit(self):
        for epoch in tqdm(range(self.epochs)):
            for (x, y) in self.dataset.dataset:
                self.train_step(x, y)
        self.generator.save('generator_model')
        self.discriminator.save('discriminator_model')

    def load_models(self):
        self.generator = tf.keras.models.load_model('generator_model')
        self.discriminator = tf.keras.models.load_model('discriminator_model')

    def generate(self, test_path, output_path):
        for img_name in os.listdir(test_path):
          print(img_name)
          img = np.expand_dims(np.array(cv2.imread(os.path.join(test_path, img_name))), axis=0)
          predicted_img = np.array(self.generator([img]))[0]
          cv2.imwrite(os.path.join(output_path, img_name), predicted_img)
          print(os.path.join(output_path, img_name))
            
