#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
np.random.seed(0)


class DataPreprocessing:
    def __init__(self,
                 IMG_HEIGHT=256,
                 IMG_WIDTH=256,
                 dataset_name='cycle_gan/vangogh2photo',
                 data_augmentation_flag=True
                 ):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.dataset_name = dataset_name
        self.data_augmentation_flag = data_augmentation_flag
        self.train_a = np.array([])
        self.train_b = np.array([])
        self.test_a = np.array([])
        self.test_b = np.array([])

    def dataset_init(self):
        '''
        This method download and transform a given dataset
        Works only for cyclegan dataset at the moment
        '''
        # Load dataset
        print('Downloading dataset....')
        dataset, metadata = tfds.load(self.dataset_name,
                                      with_info=True, as_supervised=False)

        # Transforming into np arrays
        print('Transforming dataset in np arrays....')
        self.train_a = np.asarray([self.image_resize(example['image']) for example in tfds.as_numpy(dataset['trainA'])])
        self.test_a = np.asarray([self.image_resize(example['image']) for example in tfds.as_numpy(dataset['testA'])])
        self.train_b = np.asarray([self.image_resize(example['image']) for example in tfds.as_numpy(dataset['trainB'])])
        self.test_b = np.asarray([self.image_resize(example['image']) for example in tfds.as_numpy(dataset['testB'])])

    def create_dataset(self,):
        '''
        Method to create the dataset
        '''
        self.dataset_init()
        if self.data_augmentation_flag:
            self.data_augmentation()

    def image_resize(self, image):
        return cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)

    def random_crop(self, image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return self.image_resize(crop)


    def image_edit(self, image):
        '''
        Applies some augmentation techniques
        '''
        # Mirror flip
        flipped = tf.image.flip_left_right(image)
        # Up down flip
        flip_up_down = tf.image.flip_up_down(image)
        # Transpose flip
        transposed = tf.image.transpose(image)
        # Saturation
        satured = tf.image.adjust_saturation(image, 3)
        # Brightness
        brightness = tf.image.adjust_brightness(image, 0.4)
        # Contrast
        contrast = tf.image.random_contrast(image, lower=0.0, upper=1.0)
        # Random crop
        cropped = self.random_crop(image, 128, 128)
        return [flipped, flip_up_down, transposed, satured, brightness, contrast, cropped]



    def augment_dataset(self, dataset, max_number_of_sample):
        '''
        Augment a dataset until a limit
        '''
        np.random.shuffle(dataset)
        aug_dataset = dataset
        for image in dataset:
            if aug_dataset.shape[0] >= max_number_of_sample:
                break
            for edit in self.image_edit(image):
                aug_dataset = np.vstack((aug_dataset, np.expand_dims(edit, axis=0)))
        return aug_dataset


    def data_augmentation(self, balanced_class=True):
        # Find the less populated dataset and multiply by the number of augmentation techniques
        aug_techs = 7
        max_number_of_sample_train = min(self.train_a.shape[0], self.train_b.shape[0])*aug_techs
        max_number_of_sample_test = min(self.test_a.shape[0], self.test_b.shape[0])*aug_techs

        # Augment dataset
        print("Augmenting train A.....")
        self.train_a = self.augment_dataset(self.train_a, max_number_of_sample_train)
        print("Done train A")
        print("Augmenting train B....")
        self.train_b = self.augment_dataset(self.train_b, max_number_of_sample_train)
        print("Done train B")
        print("Augmenting test A....")
        self.test_a = self.augment_dataset(self.test_a, max_number_of_sample_test)
        print("Done test A")
        print("Augmenting test B....")
        self.test_b = self.augment_dataset(self.test_b, max_number_of_sample_test)
        print("Done test B")

        # Sample
        if balanced_class:
            ind_train_vvg = np.random.choice(self.train_a.shape[0], max_number_of_sample_train, replace=False)
            self.train_a = self.train_a[ind_train_vvg]

            ind_train_photo = np.random.choice(self.train_b.shape[0], max_number_of_sample_train, replace=False)
            self.train_b = self.train_b[ind_train_photo]

            ind_test_vvg = np.random.choice(self.test_a.shape[0], max_number_of_sample_test, replace=False)
            self.test_a = self.test_a[ind_test_vvg]

            ind_test_photo = np.random.choice(self.test_b.shape[0], max_number_of_sample_test, replace=False)
            self.test_b = self.test_b[ind_test_photo]

            assert self.train_a.shape == self.train_b.shape
            assert self.test_a.shape == self.test_b.shape

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def writeTfRecord(self, output_dir):
        '''
        Method to write tfrecord
        '''
        # open the TFRecords file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        writer = tf.io.TFRecordWriter(output_dir+'train.tfrecords')
        datasets = [[self.train_a, self.train_b], [self.test_a, self.test_b]]
        for dataset in datasets:
            label = '0'
            for set in dataset:
                for img in tqdm(set):
                    # Create a feature
                    feature = {'label': self._bytes_feature(tf.compat.as_bytes(label)),
                               'image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    # Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                label = '1'
            writer.close()
            writer = tf.io.TFRecordWriter(output_dir + 'test.tfrecords')
        writer.close()

    def decode(self, serialized_example):
        """
        Parses an image and label from the given `serialized_example`.
        It is used as a map function for `dataset.map`
        """
        IMAGE_SHAPE = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)

        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
            })

        # 2. Convert the data
        image = tf.io.decode_raw(features['image'], tf.uint8)
        label = tf.strings.to_number(features['label'])

        # 3. reshape
        image = tf.convert_to_tensor(tf.reshape(image, IMAGE_SHAPE))

        return image, label


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()
    # Download and create dataset
    preprocessing_class.create_dataset()
    # Write tf record
    preprocessing_class.writeTfRecord('tfrecords/')

    # Read TfRecord
    tfrecord_path = 'tfrecords/train.tfrecords'
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse the record into tensors with map.
    # map takes a Python function and applies it to every sample.
    dataset = dataset.map(preprocessing_class.decode)

    # Divide in batch
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = iter(dataset)

    # Element of iterator
    a = iterator.get_next()

