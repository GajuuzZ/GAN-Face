import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import model_from_yaml

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout,
                             QVBoxLayout, QGridLayout, QGroupBox, QSlider,
                             QPushButton, QRadioButton, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

MODEL_FOLDER = 'Face_ACGAN_Saved/ACGAN(Adam0002 ImgL RFb ACly-GCnvTs3-DMxp-DdCnvs3)'
GENERATOR_YAML = 'generator.yaml'
GENERATOR_H5 = 'generator.h5'


def load_model():
    y_fil = os.path.join(MODEL_FOLDER, GENERATOR_YAML)
    w_fil = os.path.join(MODEL_FOLDER, GENERATOR_H5)
    with open(y_fil, 'r') as yf:
        yaml = yf.read()

    model = model_from_yaml(yaml)
    model.load_weights(w_fil)

    return model


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Nightmare-Face Generator'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500

        self.model = load_model()
        self.latent_res = 1000000
        self.res_amp = 4
        self.latent_dim = self.model.inputs[0].shape[-1].value
        self.sex_dim = False
        if len(self.model.inputs) == 2:
            self.sex_dim = True

        self.initUI()

        #img = cv2.imread('156466.jpg')
        #self.load_image(img)
        self.gen_face()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.img_label = QLabel(self)
        imbox = QHBoxLayout()
        imbox.setAlignment(Qt.AlignCenter)
        imbox.addWidget(self.img_label)

        self.grid = QGridLayout()
        self.grid.setSpacing(5)
        group_param = QGroupBox('Params')
        gvbox = QVBoxLayout()

        if self.sex_dim:
            group_sex = QGroupBox('Gender')
            m_rbt = QRadioButton('Male')
            m_rbt.setChecked(True)
            f_rbt = QRadioButton('Female')

            self.gsbox = QHBoxLayout()
            self.gsbox.addWidget(m_rbt)
            self.gsbox.addWidget(f_rbt)
            group_sex.setLayout(self.gsbox)

            m_rbt.toggled.connect(self.rbt_checked)
            f_rbt.toggled.connect(self.rbt_checked)

            gvbox.addWidget(group_sex)

        pframe = QFrame()
        pframe.setFrameShape(QFrame.StyledPanel)
        pframe.setLayout(self.grid)
        gvbox.addWidget(pframe)

        group_param.setLayout(gvbox)

        r, c = 0, 0
        for i in range(1, self.latent_dim + 1):
            sld = QSlider(Qt.Vertical, self)
            sld.setRange(-self.latent_res * self.res_amp,
                         self.latent_res * self.res_amp)
            sld.setValue(0)
            sld.setTickInterval(1)
            sld.valueChanged[int].connect(self.sld_changeValue)

            self.grid.addWidget(sld, r, c)
            c += 1
            if i % 50 == 0:
                r += 1
                c = 0

        hbox = QHBoxLayout()
        hbox.addLayout(imbox)
        hbox.addWidget(group_param)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)

        self.sld_label = QLabel(self)
        self.sld_label.setText('Slider')
        vbox.addWidget(self.sld_label)

        btn = QPushButton('Random', self)
        btn.clicked.connect(self.ok_btnClick)
        vbox.addWidget(btn)

        self.setLayout(vbox)

        self.show()

    def gen_face(self):
        latent = np.zeros((1, self.grid.count()))
        for i in range(self.grid.count()):
            sld = self.grid.itemAt(i).widget()
            latent[0][i] = sld.value() / self.latent_res

        if self.gsbox.itemAt(0).widget().isChecked():
            sex = np.array([[1]])
        else:
            sex = np.array([[0]])

        gen_img = self.model.predict([latent, sex])[0]
        gen_img = ((0.5 * gen_img + 0.5) * 255).astype('uint8')

        self.load_image(gen_img)

    def rbt_checked(self):
        self.gen_face()

    def ok_btnClick(self):
        latent = np.random.normal(0, 1, 100)
        for i, lat in enumerate(latent):
            sld = self.grid.itemAt(i).widget()
            sld.setValue(int(lat * self.latent_res))

        self.gen_face()

    def sld_changeValue(self, value):
        vl = (value / self.latent_res)
        self.sld_label.setText(str(vl))

        self.gen_face()

    def load_image(self, img_np):
        img_np = cv2.resize(img_np, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)

        h, w, c = img_np.shape
        bytesperline = c * w
        image = QImage(img_np.data, w, h, bytesperline, QImage.Format_RGB888)

        self.img_label.setPixmap(QPixmap.fromImage(image))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = App()
