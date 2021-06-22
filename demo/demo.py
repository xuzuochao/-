from PyQt5.QtWidgets import QApplication
from gui.bankCardIdentify import BaseWidget

import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BaseWidget()
    main_window.show()
    sys.exit(app.exec_())
