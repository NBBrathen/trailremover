import os
import sys

from pathlib import Path
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QStatusBar,
    QToolBar,
    QDialog,
    #QAction,
    #QWidget,
    QPushButton,
    QFileDialog
)

# global variable to show which state the toolbar is currently in
current_state = "Main_Window"

class LoadImageWindow(QDialog):
    """
    This window is displayed once the user clicks the 'Load Image'
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

        # when the browse button is clicked, browse the user's files
        self.browse.clicked.connect(self.browse_files)


        # when the upload button is clicked, close the window
        self.upload_image.clicked.connect(self.close)


    def browse_files(self):
        dir = QFileDialog.getExistingDirectory(self, "Load Image", os.path.expanduser('~')) # self, browser name, default path can also be ""
        self.filename.setText(dir)

        fits_images = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".fit") or file.endswith(".fits") or file.endswith(".fts"):
                    fits_images.append(os.path.join(root, file))
        
        for image in fits_images:
            print(image)

        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the title of the project
        self.setWindowTitle("Trail Remover")
           
        #def _createMenu(self):
        #menu = self.menuBar().addMenu("&Menu")
        #menu.addAction("&Exit", self.close)

        # Give the user a nice welcome message once they open the GUI
        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter) # move the text to the middle of the screen
        self.setCentralWidget(welcomeMsg)

        # create the menu, toolbar, and status bar
        #self._createMenu()
        self.main_state()
        self._createStatusBar()

    def main_state(self):
        
        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(welcomeMsg)
        
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.addAction("Exit", self.close)

        # add "Load Image" button, which will take the user to a new window 
        # & trigger image processing state
        load_image = QPushButton("Load Image")
        load_image.clicked.connect(self.show_new_window)

        # let the user know what this button does & add it to the toolbar
        load_image.setStatusTip("Click here to pull up the Load Image screen!")
        toolbar.addWidget(load_image)

        # add the toolbar itself!
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)

    def image_processing_state(self):
     #updating the toolbar after the user uploads their images
        toolbar = self.findChild(QToolBar)
        if toolbar:
            toolbar.clear() 
            #repopulate the toolbar with exit & return to previous
            toolbar.addAction("Exit", self.close)
            toolbar.addAction("Previous", self.main_state)
            
            # detect trails button added (currently only takes them to the next stage)
            detect_trails = QPushButton("Detect Trails")
            #detect_trails.clicked.connect(self.loading_screen)
            detect_trails.setStatusTip("Click here to start detecting the trails")
            toolbar.addWidget(detect_trails)
    

   # def loading_screen (self): 


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
        self.show_new_toolbar()

    def show_new_toolbar(self):
        global current_state
        if current_state == "Main_Window":
            self.image_processing_state()
            current_state = "Image_Processing"


    def _createStatusBar(self):
        # default status is blank: ""
        status = QStatusBar()
        self.setStatusBar(status)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setGeometry(100, 100, 1300, 750)
    window.show()
    sys.exit(app.exec())