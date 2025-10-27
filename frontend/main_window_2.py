import sys

from PyQt5.QtCore import QSize, Qt

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QAction,
    QPushButton,
)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trail Remover")

        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(welcomeMsg)

        self.setFixedSize(QSize(1300, 750))

        #self._createMenu()
        self._createToolBar()
        self._createStatusBar()

    #def _createMenu(self):
        #menu = self.menuBar().addMenu("&Menu")
        #menu.addAction("&Exit", self.close)

    def _createToolBar(self):
        toolbar = QToolBar("This is the one and only toolbar")
        # add "Exit" button
        toolbar.addAction("Exit", self.close)

        # add "Load Image" button
        button_action = QAction("Load Image", self)
        button_action.setStatusTip("Click here to pull up the Load Image screen!")
        button_action.triggered.connect(self.toolbar_button_clicked)
        toolbar.addAction(button_action)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)
    
    def toolbar_button_clicked(self, s):
        print("click", s)

    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("Not loading anything right now...")
        self.setStatusBar(status)

if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.setGeometry(100, 100, 1300, 750)
    window.show()
    sys.exit(app.exec())