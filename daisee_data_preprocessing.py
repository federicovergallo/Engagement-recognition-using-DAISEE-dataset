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
                 data_augmentation_flag=False,
                 max_frames=3
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
        self.max_frames = max_frames
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
                pictures = random.sample(pictures, self.max_frames)
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

    def resize(self, image):
        return cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)

    def face_cropping(self, image):
        # Crop and resize
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
        try:
            if faces != 0:
                x, y, w, h = faces[0]
                image = image[y:y+h, x:x+w]
        except:
            pass
        return self.resize(image)

    def random_crop(self, image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return self.face_cropping(crop)


    def augment_image(self, image):
        '''
        Applies some augmentation techniques
        '''
        # Mirror flip
        flipped = tf.image.flip_left_right(image).numpy()
        # Transpose flip
        transposed = tf.image.transpose(image).numpy()
        # Saturation
        satured = tf.image.adjust_saturation(image, 3).numpy()
        # Brightness
        brightness = tf.image.adjust_brightness(image, 0.4).numpy()
        # Contrast
        contrast = tf.image.random_contrast(image, lower=0.0, upper=1.0).numpy()
        # Resize at the end
        images = [self.resize(image) for image in [flipped, transposed, satured, brightness, contrast]]
        return images


    def get_label_picture(self, image_path, label_df):
        error_ = False
        video = image_path.split("/")[-2]
        label_series = label_df.loc[((label_df['ClipID'] == video+'.avi') | (label_df['ClipID'] == video+'.mp4'))]
        try:
            index = label_series.index.values[0]
            label = np.array([label_series['Boredom'].get(index),
                              label_series['Engagement'].get(index),
                              label_series['Confusion'].get(index),
                              label_series['Frustration '].get(index)])
            label_one_hot = (label >= 1).astype(np.uint8)
        except:
            print('Error in label picture')
            print(image_path)
            label_one_hot = ''
            error_ = True
        return label_one_hot, error_

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def writeTfRecord(self, output_dir, data_augmentation=False):
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
                img = self.face_cropping(img)
                # Read the label
                label, error_ = self.get_label_picture(image_path, label_df)
                if error_:
                    continue
                # Create a feature
                if data_augmentation:
                    images = self.augment_image(img)
                else:
                    images = img
                for image in images:
                    feature = {'label': self._bytes_feature(tf.compat.as_bytes(label.tostring())),
                               'image': self._bytes_feature(tf.compat.as_bytes(image.tostring()))}
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
        label = tf.io.decode_raw(features['label'], tf.uint8)

        # Cast
        label = tf.cast(label, tf.float32)

        # 3. reshape
        image = tf.convert_to_tensor(tf.reshape(image, IMAGE_SHAPE))

        return image, label


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()
    # Write tf recordfloat32
    preprocessing_class.writeTfRecord('tfrecords/', data_augmentation=True)

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

