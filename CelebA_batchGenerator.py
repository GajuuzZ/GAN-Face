import os
import cv2
import pandas
import numpy as np

import matplotlib.pyplot as plt


def norm_images(images):
    return (images.astype(np.float32) - 127.5) / 127.5


class CelebA:
    def __init__(self, width, height, batch_size=32, expand=(5, 5, 5, 5)):
        self.w = width
        self.h = height
        self.batch_size = batch_size
        if len(expand) == 1:
            expand = tuple([expand] * 4)
        self.ex = expand  # For crop face more widely.

        self.image_folder = './Face_data/img_align_celeba'
        self.attr_file = './Face_data/new_celeba_attr.csv'

        self.attr = pandas.read_csv(self.attr_file)
        # Get rid of wrong bbox.
        self.attr = self.attr[(self.attr['left'] > expand[0]) &
                              (self.attr['top'] > expand[1]) &
                              (self.attr['bottom'] < 218 - expand[2]) &
                              (self.attr['right'] < 178 - expand[3])].reset_index(drop=True)

    def __len__(self):
        return int(np.floor(len(self.attr) / self.batch_size))

    def shuffle_data(self):
        self.attr = self.attr.sample(frac=1).reset_index(drop=True)

    def load_img(self, file_name, bbox=None):
        img = cv2.imread(os.path.join(self.image_folder, file_name))
        if bbox is not None:
            img = img[bbox[1] - self.ex[1]:bbox[3] + self.ex[2], bbox[0] - self.ex[0]:bbox[2] + self.ex[3]]

        img = cv2.resize(img, (self.w, self.h), cv2.INTER_AREA)
        # Convert to RGB.
        return img[:, :, ::-1]

    def get_batch_gender(self, index):
        batch = self.attr.iloc[index * self.batch_size:(index + 1) * self.batch_size, :].reset_index(drop=True)
        batch_x = np.zeros((self.batch_size, self.h, self.w, 3), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, 1))

        for i, row in batch.iterrows():
            img = self.load_img(row[0], row[-4:])
            gender = 0 if row['Male'] == -1 else 1

            batch_x[i, :] = img
            batch_y[i, :] = gender

        batch_x = norm_images(batch_x)
        return batch_x, batch_y

    def show_sample_images(self, num):
        random_idx = np.random.choice(len(self.attr), num)
        n = int(np.ceil(np.sqrt(num)))

        fig = plt.figure()
        for i, idx in enumerate(random_idx):
            ax = fig.add_subplot(n, n, i + 1)
            img = self.load_img(self.attr.iloc[idx, 0],
                                self.attr.iloc[idx, -4:])

            ax.imshow(img)
            ax.axis('off')

        plt.show()
