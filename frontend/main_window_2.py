import sys

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

class LoadImageWindow(QWidget):
    """
    This window is displayed once the user clicks the'Load Image'
    button on the main window. It allows the user to add FITS images
    into the program to be used later on.
    """
    def __init__(self):
        super(LoadImageWindow, self).__init__()
        loadUi("load_image.ui", self)

        #layout = QVBoxLayout()
        #self.label = QLabel("Load Image window")
        #layout.addWidget(self.label)
        #self.setLayout(layout)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the title and size of the project
        self.setWindowTitle("Trail Remover")
        #self.setFixedSize(QSize(1300, 750))

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
        toolbar.addAction("Exit", self.close)

        # add "Load Image" button, which will take the user to a new window
        load_image = QAction("Load Image", self)

        # let the user know what this button does
        load_image.setStatusTip("Click here to pull up the Load Image screen!")
        
        # activate the button
        load_image.triggered.connect(self.toolbar_button_clicked)
        load_image.setCheckable(True)
        toolbar.addAction(load_image)

        # open a new window once the button is clicked
        load_image_2 = QPushButton("Load_Image_2.0")
        load_image_2.clicked.connect(self.show_new_window)
        #toolbar.addAction(load_image_2)
        toolbar.addWidget(load_image_2)

        # add the toolbar!
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)
    
    def toolbar_button_clicked(self, s):
        # update the program when a toolbar button is clicked
        print("click", s)

    def show_new_window(self, checked):
        if self.window is None:
            self.window = LoadImageWindow()
            self.window.show()
        else:
            #self.widow.close()
            self.window = None

    def _createStatusBar(self):
        status = QStatusBar()
        
        # add a default message to the status bar
        status.showMessage("Not loading anything right now...")

        # add the status bar!
        self.setStatusBar(status)

if __name__ == "__main__":
    #app = QApplication([])
    #window = Window()
    #window.setGeometry(100, 100, 1300, 750)
    #window.show()

    app = QApplication(sys.argv)
    load_image_window = LoadImageWindow()
    widget = QtWidgets.QtStackedWidget()
    widget.addWidget(load_image_window)
    widget.setFixedWidth(400)
    widget.setFixedHeight(300)
    widget.show()

    sys.exit(app.exec())