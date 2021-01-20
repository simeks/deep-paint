import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import os
from collections import deque

import numpy as np
import SimpleITK as sitk

from net import Network

class Subject(object):
    def __init__(self, file):
        self.file = file
        self.segm_file = self.file[:-4] + '_segm.vtk'
        self.sitk_img = sitk.ReadImage(file)
        self.img = sitk.GetArrayFromImage(self.sitk_img)
        self.img = self.img / np.max(self.img)
        self.img = self.img[self.img.shape[0]//2:self.img.shape[0]//2+1]
        
        if os.path.isfile(self.segm_file):
            img = sitk.ReadImage(self.segm_file)
            self.segm = sitk.GetArrayFromImage(img)
        else:
            self.segm = None

    def save(self):
        pass
        #img = sitk.GetImageFromArray(self.segm)
        #img.CopyInformation(self.sitk_img)
        #sitk.WriteImage(img, self.segm_file)


class SubjectView(object):
    def __init__(self, subject):
        self.subject = subject
        self.initialize()

    def initialize(self):
        self.slices = []
        self.segms = []
        
        img = self.subject.img
        segm = self.subject.segm

        shape = img.shape
        img_max = np.max(img)
        for z in range(shape[0]):
            qi = QImage((255*img[z]/img_max).astype(np.uint8), shape[2], shape[1], shape[2], QImage.Format_Grayscale8)
            self.slices.append(qi)

            qi = QImage(segm[z].astype(np.uint8), shape[2], shape[1], shape[2], QImage.Format_Alpha8)
            self.segms.append(qi)
        

    def save(self):
        """ Write changes (in QImage) to numpy array """

        shape = self.subject.img.shape
        
        for z in range(shape[0]):
            segm = self.segms[z]

            # Seems like we cant trust that QImage.bytesPerLine() == BPP*QImage.width()
            #   so we'll need to copy line by line
            for y in range(segm.height()):
                b = segm.constScanLine(y)
                b.setsize(segm.width())

                self.subject.segm[z,y] = np.array(b)
        self.subject.save()

    def clear(self):
        """ Clears the segmentation completely """

        self.subject.segm = np.zeros(self.subject.segm.shape, dtype=np.uint8)
        self.initialize()



class DataLoader(object):
    def __init__(self, files, buffer_size=30):
        self.files = files
        self.next_idx = 0
        self.buffer = deque(maxlen=buffer_size)

    def next(self):
        if self.next_idx >= len(self.files):
            return

        f = self.files[self.next_idx]
        self.next_idx += 1

        subject = Subject(f)
        self.buffer.append(subject)

        return subject



class App(QWidget):
    DRAW_NONE = 0
    DRAW_ADD = 1
    DRAW_REMOVE = 2

    def __init__(self, files):
        super().__init__()
        self.network = Network()
        self.loader = DataLoader(files)
        self.subject_view = None
        self.num_trained = 0
        self.previous = []
        self.draw_mode = App.DRAW_NONE
        self.setGeometry(100, 100, 500, 500)
        self.path = None

        self.next()

    def next(self):
        self.save()
        
        subject = self.loader.next()
        if not subject:
            return

        if subject.segm is None:
            if self.num_trained < 1:
                subject.segm = np.zeros(subject.img.shape, dtype=np.uint8)
            else:
                subject.segm = (255*self.network.predict(subject.img)).astype(np.uint8)

        self.subject_view = SubjectView(subject)
        self.slice_index = len(self.subject_view.slices)//2

        self.update()

    def save(self):
        if not self.subject_view:
            return

        self.subject_view.save()

        subject = self.subject_view.subject

        amax = np.max(subject.segm.reshape(subject.segm.shape[0], -1), axis=1)
        ids = np.where(amax > 0)[0]
        if len(ids) == 0:
            return # No segmentation

        zrange = (ids[0], ids[-1]+1)

        self.previous.append((
            subject.img, subject.segm, zrange
        ))

        imgs = []
        segms = []

        for p in self.previous[-5:]:
            for z in range(p[2][0], p[2][1]):
                imgs.append(p[0][z:z+1].astype(np.float32))
                segms.append((p[1][z:z+1] == 255).astype(np.long))

        self.network.fit(imgs, segms)
        self.num_trained += 1

    def clear(self):
        if not self.subject_view:
            return

        self.subject_view.clear()
        self.update()


    def paintEvent(self, event):
        if not self.subject_view:
            return
        
        painter = QPainter(self)

        pixmap = QPixmap(self.subject_view.slices[self.slice_index])
        painter.drawPixmap(self.rect(), pixmap)

        pixmap = QPixmap(self.subject_view.segms[self.slice_index])

        painter2 = QPainter(pixmap)
        painter2.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter2.fillRect(pixmap.rect(), QColor('blue'))
        painter2.end()

        painter.setOpacity(0.5)
        painter.drawPixmap(self.rect(), pixmap)
        painter.setOpacity(1)

        if self.path:
            color = QColor('white')
            if self.draw_mode == App.DRAW_ADD:
                color = QColor('green')
            elif self.draw_mode == App.DRAW_REMOVE:
                color = QColor('red')

            painter.setPen(QPen(color, 2))
            painter.drawPath(self.path)

    def wheelEvent(self,event):
        d = int(event.angleDelta().y()) // 120

        self.slice_index = max(0, min(self.slice_index + d, len(self.subject_view.slices)-1))
        self.update()

    def mousePressEvent(self, event):
        mod = event.modifiers()
        if mod & Qt.ShiftModifier:
            self.draw_mode = App.DRAW_REMOVE
        else:
            self.draw_mode = App.DRAW_ADD

        self.path = QPainterPath()
        self.path.moveTo(event.pos())
    
    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        self.update()
    
    def mouseReleaseEvent(self, event):
        self.path.closeSubpath()
        
        segm = self.subject_view.segms[self.slice_index]

        painter = QPainter(segm)
        painter.setBrush(Qt.white)

        # Resample path to image space
        transform = QTransform()
        transform.scale(
            segm.width() / self.rect().width(),
            segm.height() / self.rect().height()
        )
        painter.setTransform(transform)
        
        if self.draw_mode == App.DRAW_REMOVE:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        
        painter.fillPath(self.path, QColor('white'))
        painter.end()

        self.path = None
        self.draw_mode = App.DRAW_NONE
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.app.quit()
        elif key == Qt.Key_S:
            self.save()
        elif key == Qt.Key_N:
            self.next()
        elif key == Qt.Key_C:
            self.clear()



images = [
    'images/a01.vtk',
    'images/a02.vtk',
    'images/a03.vtk',
    'images/a04.vtk',
    'images/a05.vtk',
    'images/a06.vtk',
    'images/a07.vtk',
    'images/a08.vtk',
    'images/a09.vtk',
    'images/a10.vtk',
    'images/a11.vtk',
    'images/a12.vtk',
    'images/a13.vtk',
    'images/a14.vtk',
    'images/a15.vtk',
    'images/a16.vtk',
    'images/a17.vtk',
    'images/a18.vtk',
    'images/a19.vtk',
    'images/a20.vtk',
    'images/a21.vtk',
    'images/a22.vtk',
    'images/a23.vtk',
    'images/a24.vtk',
    'images/a25.vtk',
    'images/a26.vtk',
    'images/a27.vtk',
    'images/a28.vtk',
    'images/a29.vtk',
    'images/a30.vtk'
]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App(images)
    window.show()
    sys.exit(app.exec_())