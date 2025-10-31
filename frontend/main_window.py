import os
import sys

from pathlib import Path
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QAction,
    QPushButton,
    QVBoxLayout,
    QDialog,
    QFileDialog
)

class LoadImageWindow(QDialog):
    """
    This window is displayed once the user clicks the'Load Image'
    button on the main window. It allows the user to add FITS images
    into the program to be used later on.
    """
    def __init__(self):
        super(LoadImageWindow, self).__init__()

        self.setWindowTitle("Load Image")
        self.setFixedWidth(400)
        self.setFixedHeight(300)

        ui_path = Path(__file__).parent / "load_image.ui"
        loadUi(str(ui_path), self)
        self.browse.clicked.connect(self.browse_files)

    def browse_files(self):
        fname = QFileDialog.getOpenFileName(self, 'Load Image', os.path.expanduser('~'), 'FITS Images (*.fit)') # self, name, path, filetype
        self.filename.setText(fname[0])

        #layout = QVBoxLayout()
        #self.label = QLabel("Load Image window")
        #layout.addWidget(self.label)
        #self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the title and size of the project
        self.setWindowTitle("Trail Remover")

        # Give the user a nice welcome message as soon as they open the GUI
        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter) # move to the middle of the screen
        self.setCentralWidget(welcomeMsg)

        # create the menu, toolbar, and status bar
        #self._createMenu()
        self._createToolBar()
        self._createStatusBar()

    #def _createMenu(self):
        #menu = self.menuBar().addMenu("&Menu")
        #menu.addAction("&Exit", self.close)

    def _createToolBar(self):
        toolbar = QToolBar("This is the one and only toolbar")

        # add "Exit" button, which will exit the program when clicked
        exit_button = toolbar.addAction("Exit", self.close)
        exit_button.setStatusTip("Clicking this button will exit the program. Are you sure?")

        # add "Load Image" button, which will take the user to a new window
        load_image = QPushButton("Load Image")
        load_image.clicked.connect(self.show_new_window)

        # let the user know what this button does & add it to the toolbar
        load_image.setStatusTip("Click here to pull up the Load Image screen!")
        toolbar.addWidget(load_image)

        # add the toolbar itself!
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)

    def show_new_window(self):
        dialog = LoadImageWindow()
        dialog.exec_()

    def _createStatusBar(self):
        status = QStatusBar()
        
        # add a default message to the status bar
        status.showMessage("")

        # add the status bar!
        self.setStatusBar(status)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setGeometry(100, 100, 1300, 750)
    window.show()

    """
    app = QApplication(sys.argv)
    load_image_window = LoadImageWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(load_image_window)
    widget.setFixedWidth(400)
    widget.setFixedHeight(300)
    widget.show()
    """

    sys.exit(app.exec())