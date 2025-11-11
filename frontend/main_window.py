import os
import sys

from pathlib import Path
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QStatusBar,
    QToolBar,
    QDialog,
    QPushButton,
    QFileDialog, 
    QSplashScreen
)

# global variable to show which state the toolbar is currently in
current_state = "Main_Window"
# global variable to hold all the images from the folder the user inputs
fits_images = []

class LoadingScreen(QSplashScreen):
    def __init__(self):
        super(QSplashScreen, self).__init__()
        ui_path = Path(__file__).parent / "loading_screen.ui"
        loadUi(str(ui_path),self)


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
        # open the user's home directory
        dir = QFileDialog.getExistingDirectory(self, "Load Images", os.path.expanduser('~')) # self, browser name, default path can also be ""
        self.filename.setText(dir)

        global fits_images
        for root, _, files in os.walk(dir):
            for file in files:
                # only add files to fits_images if they are actally fits images!!!
                if file.endswith(".fit") or file.endswith(".fits") or file.endswith(".fts"):
                    fits_images.append(os.path.join(root, file))
        
        # temporary print to show all images in fits_images
        for image in fits_images:
            print(image)

        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the title of the project
        self.setWindowTitle("Trail Remover")

        # Give the user a nice welcome message once they open the GUI
        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter) # move the text to the middle of the screen
        self.setCentralWidget(welcomeMsg)

        # create the menu, toolbar, status bar, and splash/loading screen
        #self._createMenu()
        self.toolbar = QToolBar("Main Toolbar")
        self.main_state()
        self._createStatusBar()
        self.loading_screen = LoadingScreen()

    #def _createMenu(self):
        #menu = self.menuBar().addMenu("&Menu")
        #menu.addAction("&Exit", self.close)

    def main_state(self):
        welcomeMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>")
        welcomeMsg.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(welcomeMsg)
        
        # clear toolbar & repopulate it
        self.toolbar.clear()
        self.toolbar.addAction("Exit", self.close)

        # add "Load Images" button, which will take the user to a new window & trigger image processing state
        load_images = QPushButton("Load Images")
        load_images.clicked.connect(self.show_new_window)

        # let the user know what this button does & add it to the toolbar
        load_images.setStatusTip("Click here to pull up the Load Images screen!")
        self.toolbar.addWidget(load_images)

        # add the toolbar itself!
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

    def display_images(self, file_path):
        # TODO: upload MULTIPLE files
        # TODO: change fits images to an appropriate format?

        # get the file that the user uploaded
        pixmap = QPixmap(file_path)

        #turn the pixmap into a label
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # update the central widget
        self.setCentralWidget(image_label)

    def image_processing_state(self):
        # update the central widget after the user uploads their images
        print("Central Widget Image:" + fits_images[0])
        self.display_images(fits_images[0])

        # update the toolbar after the user uploads their images
        if self.toolbar:
            self.toolbar.clear() 
            # repopulate the toolbar with exit & return to previous
            self.toolbar.addAction("Exit", self.close)

            # previous button that will clear the toolbar and send you back to the original page
            prev_button = QPushButton("Previous")
            direction = "backwards"
            prev_button.clicked.connect(lambda: self.show_new_toolbar_image_processing(direction))
            self.toolbar.addWidget(prev_button)
            prev_button.setStatusTip("Click here to go back to the previous state (the Main Window)!")
            
            # upon selecting detect trails it shows the users the loading screen 
            detect_trails = QPushButton("Detect Trails")
            detect_trails.clicked.connect(self.show_loading_screen)
            detect_trails.setStatusTip("Click here to start detecting trails in your fits images")
            self.toolbar.addWidget(detect_trails)

   # def loading_screen (self): 

    def show_new_window(self):
        dialog = LoadImageWindow()
        dialog.exec_()
        self.show_new_toolbar_main()

    def show_new_toolbar_main(self):
        global current_state
        if current_state == "Main_Window":
            self.image_processing_state()
            current_state = "Image_Processing"

    def show_new_toolbar_image_processing(self, direction):
        global current_state
        if current_state == "Image_Processing":
            toolbar = self.findChild(QToolBar)
            if direction == "forward":
                for widget in toolbar.actions():
                    toolbar.removeAction(widget)
                #self.detection_state()
                current_state = "Detection"
            elif direction == "backwards": # should be an else statement?
                for widget in toolbar.actions():
                    toolbar.removeAction(widget)
                self.main_state()
                current_state = "Main_Window"
                

    def _createStatusBar(self):
        # default status is blank: ""
        status = QStatusBar()
        self.setStatusBar(status)

    def show_loading_screen(self):
        # shows loading screen (default is 24% idk why)
        self.loading_screen.show()

        # for now it just shows the screen for 5 seconds
        QTimer.singleShot(5000, self.loading_screen.close)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setGeometry(100, 100, 1300, 750)
    window.show()
    sys.exit(app.exec())