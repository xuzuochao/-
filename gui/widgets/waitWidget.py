from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QMovie


class WaitWidget(QWidget):
    def __init__(self, parent=None, objectName="WaitWidget"):
        super(WaitWidget, self).__init__(parent)
        self.setObjectName(objectName)
        self.setStyleSheet("background-color:#FFFFFF")
        hLayout = QHBoxLayout()
        self.movie = QMovie(":/image/loading.gif")
        ## 用于居中的label
        self.label1 = QLabel()
        self.label2 = QLabel()
        ## 存放gif的label
        self.gifLabel = QLabel()
        self.gifLabel.setFixedWidth(100)
        self.gifLabel.setMovie(self.movie)
        hLayout.setSpacing(0)  # 内部间距0
        hLayout.setContentsMargins(0, 0, 0, 0)  # 外边框的距离0
        hLayout.addWidget(self.label1)
        hLayout.addWidget(self.gifLabel)
        hLayout.addWidget(self.label2)
        self.setLayout(hLayout)
