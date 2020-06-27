#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import tensorflow_datasets as tfds
import pandas as pd
import random


AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
np.random.seed(0)


class DataPreprocessing:
    def __init__(self,
                 IMG_HEIGHT=224,
                 IMG_WIDTH=224,
                 dataset_dir='dataset/DAiSEE/DataSet/',
                 test_dir='Test/',
                 train_dir='Train/',
                 val_dir='Validation/',
                 labels_dir='dataset/DAiSEE/Labels/',
                 test_label='TestLabels.csv',
                 train_label='TrainLabels.csv',
                 val_label='ValidationLabels.csv',
                 data_augmentation_flag=False
                 ):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.dataset_dir = dataset_dir
        self.train_dir = self.dataset_dir+train_dir
        self.test_dir = self.dataset_dir+test_dir
        self.val_dir = self.dataset_dir+val_dir
        self.labels_dir = labels_dir
        self.train_label_dir = self.labels_dir + train_label
        self.test_label_dir = self.labels_dir + test_label
        self.val_label_dir = self.labels_dir + val_label
        self.data_augmentation_flag = data_augmentation_flag
        self.face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

    def get_images_from_set_dir(self, setdir):
        '''
        Method to find all images in the tree folder
        '''
        set_dir_images = []
        humans = os.listdir(setdir)
        for human in humans:
            human_dir = setdir + human + "/"
            videos = os.listdir(human_dir)
            for video in videos:
                video_dir = human_dir + video + "/"
                pictures = os.listdir(video_dir)
                pictures = random.sample(pictures, 10)
                for picture in pictures:
                    picture_dir = video_dir + picture
                    if picture.endswith(".jpg"):
                        set_dir_images.append(picture_dir)
        return set_dir_images

    def get_labels_dataframe(self):
        '''
        Method to read pandas dataframe
        '''
        train_df = pd.read_csv(self.train_label_dir, sep=",")
        test_df = pd.read_csv(self.test_label_dir, sep=",")
        val_df = pd.read_csv(self.val_label_dir, sep=",")
        return train_df, test_df, val_df

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
        # Crop and resize
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
        try:
            if faces != 0:
                x, y, w, h = faces[0]
                image = image[y:y+h, x:x+w]
        except:
            pass
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

    def get_label_picture(self,image_path, label_df):
        error_ = False
        video = image_path.split("/")[-2]
        label_series = label_df.loc[label_df['ClipID'] == video+'.avi']
        try:
            index = label_series.index.values[0]
            label = np.array([label_series['Boredom'].get(index),
                              label_series['Engagement'].get(index),
                              label_series['Confusion'].get(index),
                              label_series['Frustration '].get(index)])/4
        except:
            print('Error in label picture')
            print(image_path)
            label = ''
            error_ = True
        return label, error_



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

        # Read dataframes
        train_df, test_df, val_df = self.get_labels_dataframe()

        # Objects to iterate
        objs = [('train', self.train_dir, train_df),
                ('test', self.test_dir, test_df),
                ('val', self.val_dir, val_df)]

        for name, dataset, label_df in tqdm(objs):
            # Open Writer
            writer = tf.io.TFRecordWriter(output_dir+name+'.tfrecords')
            # Get all the images of a set
            images_path = self.get_images_from_set_dir(dataset)
            for image_path in tqdm(images_path, total=len(images_path)):
                # Read the image from path
                img = cv2.imread(image_path)[..., ::-1]
                img = self.image_resize(img)
                # Read the label
                label, error_ = self.get_label_picture(image_path, label_df)
                if error_:
                    continue
                # Create a feature
                feature = {'label': self._bytes_feature(tf.compat.as_bytes(label.tostring())),
                           'image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
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
        label = tf.io.decode_raw(features['label'], tf.float64)

        label = tf.cast(label, tf.float32)

        # 3. reshape
        image = tf.convert_to_tensor(tf.reshape(image, IMAGE_SHAPE))

        return image, label


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()
    # Download and create dataset
    #preprocessing_class.create_dataset()
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

