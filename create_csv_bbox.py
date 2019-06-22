import os
import cv2
import dlib
import pandas
import numpy as np

import matplotlib.pyplot as plt

new_attr_csv = './Face_data/new_celeba_attr.csv'

image_folder = './Face_data/img_align_celeba'
attr_file = './Face_data/list_attr_celeba.csv'
bbox_file = './Face_data/list_bbox_celeba.csv'
attr_df = pandas.read_csv(attr_file)
bbox_df = pandas.read_csv(bbox_file)

# Create new dataframe from attr_df and add bbox columns to it.
new_df = pandas.DataFrame(columns=list(attr_df.columns) +
                                  ['left', 'top', 'right', 'bottom'])

# Get dlib face detector.
detector = dlib.get_frontal_face_detector()

for i, row in bbox_df.iterrows():
    print(i, row[0])
    file_path = os.path.join(image_folder, row[0])
    image = cv2.imread(file_path)

    # Detect face in image.
    faces = detector(image, 0)
    if len(faces) > 0:
        # Get largest face.
        largest_face = np.argmax([face.area() for face in faces])
        bb = faces[largest_face]

        # Create new row.
        new_row = pandas.Series(attr_df.iloc[i, :])
        new_row.loc['left'] = bb.left()
        new_row.loc['top'] = bb.top()
        new_row.loc['right'] = bb.right()
        new_row.loc['bottom'] = bb.bottom()

        new_df = new_df.append(new_row)

        # face = image[bb.top():bb.bottom(), bb.left():bb.right()]
        # plt.imshow(face[:, :, ::-1])

    # if i == 10:
    #     break

new_df.to_csv(new_attr_csv, index=False)

print('Old csv len : ', len(attr_df))
print('New csv len : ', len(new_df))
print('Face not found : ', len(attr_df) - len(new_df))
