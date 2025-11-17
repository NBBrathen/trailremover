import os
import sys

# imports for collecting fits images
from pathlib import Path
import time

# imports for formatting fits images
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np

# imports for GUI features
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QStatusBar,
    QToolBar,
    QDialog,
    QPushButton,
    QFileDialog, 
    QSplashScreen,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QScrollArea
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

    def progress(self):
        # from 0% to 100%...
        for i in range(1, 101):
            # update by 1% every 0.05s
            time.sleep(0.05)
            self.progressBar.setValue(i)
            QApplication.processEvents()

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

    def display_image(self, file_path):
        # TODO: fits images dont display as viewed in other software?
        # read the file that the user uploaded
        hdul = fits.open(file_path)
        data = hdul[0].data.astype(np.float32)
        hdul.close()

        low = np.percentile(data, 1)
        high = np.percentile(data, 99)
        data = np.clip(data, low, high)

        data_norm = (data - low) / (high - low)
        data_8bit = (data_norm * 255).astype(np.uint8)

        # get width & height and scale down
        height, width = data_8bit.shape
        new_width = width // 6
        new_height = height // 6
        data_scaled = cv2.resize(data_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)

        bytes_per_line = new_width

        # transform QImage to QPixmap
        q_image = QImage(data_scaled.data, new_width, new_height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        #turn the pixmap into a label
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # TODO: only the first clicked button will update the widget... fix that
        # update the widgets
        original_item = self.child_layout.takeAt(0)
        if original_item:
            widget = original_item.widget()
            if widget:
                # remove it from the layout
                self.child_layout.removeWidget(widget)
                widget.deleteLater() 
            del original_item # delete the old layout item

        # insert the new image
        self.child_layout.insertWidget(0, image_label)

        #TODO: also insert the 2nd image after detecting trails

    def image_processing_state(self):
        # Create the main (parent) layout & add scroll area
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)
        self.scroll_area = QScrollArea()
        self.main_layout.addWidget(self.scroll_area)
        
        self.scrollAreaContent = QWidget()
        self.scroll_area.setWidgetResizable(True)
        #self.scrollAreaContent.setGeometry(QRect(0, 0, 1225, 932))
        self.scroll_area.setWidget(self.scrollAreaContent)

        # Create child layouts
        self.scroll_layout = QVBoxLayout(self.scrollAreaContent)
        self.child_layout = QHBoxLayout()

        # add a button for each image
        #global fits_images
        for image in fits_images:
            image_button = QPushButton(image + "           Trails Detected: 0")
            image_button.setStyleSheet("text-align: left;") 
            # update the central widget after the user uploads their images
            image_button.clicked.connect(lambda checked, current_image=image: self.display_image(current_image))
            #image_button.clicked.connect(lambda: self.display_image(image))
            image_button.setStatusTip("Click here display the selected image.")
            self.scroll_layout.addWidget(image_button)
            #self.main_layout.addWidget(image_button)

        self.child_layout.addWidget(QLabel("Original Image"))
        self.child_layout.addWidget(QLabel("New Image"))

        # Add the child layout to the main layout
        self.main_layout.addLayout(self.child_layout)

        # set the whole layout as the central widget
        self.setCentralWidget(central_widget)

        # update the toolbar after the user uploads their images
        if self.toolbar:
            self.toolbar.clear() 
            # repopulate the toolbar with exit & return to previous
            self.toolbar.addAction("Exit", self.close)

            # previous button that will clear the toolbar and send you back to the original page
            prev_button = QPushButton("Previous")
            prev_button.clicked.connect(self.show_new_toolbar_main)
            self.toolbar.addWidget(prev_button)
            prev_button.setStatusTip("Click here to go back to the Main Window.")

            # after clicking save, it will save the new images for the user 
            save_button = QPushButton("Save")
            save_button.setStatusTip("Click here to save your new fits images.")
            self.toolbar.addWidget(save_button)

    def show_new_window(self):
        dialog = LoadImageWindow()
        dialog.exec_()
        self.show_new_toolbar_main()

    def show_new_toolbar_main(self):
        global current_state
        if current_state == "Main_Window":
            # change state
            self.image_processing_state()
            current_state = "Image_Processing"

            # pop up loading screen after uploading images
            self.show_loading_screen()

        elif current_state == "Image_Processing":
            toolbar = self.findChild(QToolBar)
            # reset toolbar and central widget back to main state
            for widget in toolbar.actions():
                toolbar.removeAction(widget)
            self.main_state()
            current_state = "Main_Window"

    def _createStatusBar(self):
        # default status is blank: ""
        status = QStatusBar()
        self.setStatusBar(status)

    def show_loading_screen(self):
        # shows loading screen
        self.loading_screen.show()

        # update the progress bar, and close 0.5s after getting to 100%
        self.loading_screen.progress()
        QTimer.singleShot(500, self.loading_screen.close)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setGeometry(100, 100, 1300, 750)
    window.show()
    sys.exit(app.exec())