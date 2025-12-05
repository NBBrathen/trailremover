import os
import sys
from api_client import create_api_client

# imports for collecting fits images
from pathlib import Path
import shutil

# imports for formatting fits images
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np

# imports for GUI features
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
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
# global variable to hold all the processed images
processed_images = []


class UploadAndDetectThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, client, image_paths, main_window):
        super().__init__()
        self.client = client
        self.image_paths = image_paths
        self.main_window = main_window

    def run(self):
        total_images = len(self.image_paths)
        job_ids = []

        for idx, path in enumerate(self.image_paths, 1):
            job_id = self.client.upload_image(Path(path))
            if job_id:
                status = self.client.get_job_status(job_id)
                self.main_window.image_data[job_id] = {
                    "original_path": Path(path),
                    "status": status
                }
                job_ids.append(job_id)

                # wait until status is DONE
                while True:
                    status = self.client.get_job_status(job_id)
                    backend_status = status.get("status")
                    if backend_status == "AWAITING_REVIEW":
                        detections = self.client.get_detections(job_id)
                        if detections:
                            self.main_window.image_data[job_id]["trail_count"] = detections.get("trail_count", 0)
                            trail_ids = [t["trail_id"] for t in detections["trails"]]
                            self.client.submit_corrections(job_id, trail_ids)
                    if backend_status == "DONE":
                        break

            # emit progress
            percent = int((idx / total_images) * 100)
            self.progress.emit(percent)

        self.finished.emit()


class LoadingScreen(QSplashScreen):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowStaysOnTopHint) # necessary?
        ui_path = Path(__file__).parent / "loading_screen.ui"
        loadUi(str(ui_path), self)

        self.progressBar.setValue(0)
        self.main_window = None

    def start(self, main_window, worker_thread):
        self.main_window = main_window
        self.show()
        QApplication.processEvents()

        worker_thread.progress.connect(self.progressBar.setValue)
        worker_thread.finished.connect(self.done)

    def done(self):
        self.close()
        print("All jobs completed!")
        self.main_window.image_processing_state() 


class LoadImageWindow(QDialog):
    """
    This window is displayed once the user clicks the 'Load Image'
    button on the main window. It allows the user to add FITS images
    into the program to be used later on.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent

        self.setWindowTitle("Load Image")
        self.setFixedWidth(400)
        self.setFixedHeight(300)

        ui_path = Path(__file__).parent / "load_image.ui"
        loadUi(str(ui_path), self)

        # when the browse button is clicked, browse the user's files
        self.browse.clicked.connect(self.browse_files)

        # when the upload button is clicked, close the window
        self.upload_image.clicked.connect(self.upload_images)

    def browse_files(self):
        # open the user's home directory
        dir = QFileDialog.getExistingDirectory(self, "Load Images", os.path.expanduser('~')) # self, browser name, default path can also be ""
        self.filename.setText(dir)

        global fits_images
        for root, _, files in os.walk(dir):
            for file in files:
                # only add files to fits_images if they are actally fits images!!!
                if file.endswith((".fit", ".fits", ".fts")):
                    fits_images.append(os.path.join(root, file))

    def upload_images(self):
        if fits_images is None:
            print("No images selected.")
            return
        
        self.close()

        self.parent_window.loading_screen.show()
        QApplication.processEvents()

        # start combined thread
        self.worker_thread = UploadAndDetectThread(self.parent_window.client, fits_images, self.parent_window)
        self.parent_window.loading_screen.start(self.parent_window, self.worker_thread)
        self.worker_thread.start()  


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
        self.toolbar = QToolBar("Main Toolbar")
        self.main_state()
        self._createStatusBar()
        self.loading_screen = LoadingScreen()

        # create an instance of a client
        self.client = create_api_client()
        
        # dictionary to store info for each iamge: key - job_id, 
        # value - original path, status, path after detecton
        self.image_data = {}

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

    def display_image(self, job_id):
        info = self.image_data[job_id]
        file_path = info["original_path"]

        try:
            # read the file that the user uploaded
            hdul = fits.open(file_path)
            data = hdul[0].data.astype(np.float32)
            hdul.close()
        except Exception as e:
            print(f"Error reading FITS file: {file_path} -> {e}")
            return

        low = np.percentile(data, 1)
        high = np.percentile(data, 99)
        data = np.clip(data, low, high)

        data_norm = (data - low) / (high - low)
        data_8bit = (data_norm * 255).astype(np.uint8)

        # scale down image
        height, width = data_8bit.shape
        new_width = width // 6
        new_height = height // 6
        data_scaled = cv2.resize(data_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width

        # transform QImage to QPixmap
        q_image = QImage(data_scaled.data, new_width, new_height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        #turn the pixmap into a label
        original_image_label = QLabel()
        original_image_label.setPixmap(pixmap)
        original_image_label.setAlignment(Qt.AlignCenter)

        processed_image_label = None
        if "processed_png" not in info:
            processed_path = Path(f"/tmp/{job_id}_processed.fits")
            success = self.client.download_result(job_id, processed_path)
            if success:
                # add the image to the list
                processed_images.append(processed_path)
                
                # convert from fits to png
                processed_png_path = Path(f"/tmp/{job_id}_processed.png")

                hdul = fits.open(processed_path)
                data = hdul[0].data.astype(np.float32)
                hdul.close()

                low = np.percentile(data, 1)
                high = np.percentile(data, 99)
                data = np.clip(data, low, high)

                plt.imshow(data, cmap='gray') # Having origin='lower' flips the image
                plt.axis('off')
                plt.savefig(processed_png_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                info["processed_png"] = processed_png_path
            else:
                info["processed_png"] = None
        if info.get("processed_png"):
            processed_pixmap = QPixmap(str(info["processed_png"]))
            processed_image_label = QLabel()
            processed_image_label.setPixmap(processed_pixmap)
            processed_image_label.setAlignment(Qt.AlignCenter)

        # update the widgets
        old_original_item = self.child_layout.takeAt(0)
        if old_original_item:
            original_widget = old_original_item.widget()
            if original_widget:
                # remove it from the layout
            #    self.child_layout.removeWidget(original_widget)
                original_widget.deleteLater() 
            #del old_original_item # delete the old layout item

        old_processed_item = self.child_layout.takeAt(0)
        if old_processed_item:
            processed_widget = old_processed_item.widget()
            if processed_widget:
                # remove it from the layout
                processed_widget.deleteLater() 

        # insert the new image
        self.child_layout.addWidget(original_image_label)
        if processed_image_label:
            self.child_layout.addWidget(processed_image_label)


    def image_processing_state(self):
        # Create the main (parent) layout & add scroll area
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)
        self.scroll_area = QScrollArea()
        self.main_layout.addWidget(self.scroll_area)
        
        self.scrollAreaContent = QWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scrollAreaContent)

        # Create child layouts
        self.scroll_layout = QVBoxLayout(self.scrollAreaContent)
        self.child_layout = QHBoxLayout()

        # add a button for each image
        for job_id, info in self.image_data.items():
            original_path = info["original_path"]
            trail_count = info.get("trail_count", 0)  # Get from image_data, not status

            # Build the button text
            button_text = f"{original_path}\t\t\t\t\t Trails Detected: {trail_count}"
            image_button = QPushButton(button_text)
            image_button.setStyleSheet("text-align: left;")

            # Connect the button to display the image; pass job_id so you can access processed image later
            # update the central widget after the user uploads their images
            image_button.clicked.connect(lambda checked, current_job_id=job_id: self.display_image(current_job_id))

            # TODO: try to move self.client.download_result (from self.display_images) to right here?

            image_button.setStatusTip("Click here to display the selected image.")
            self.scroll_layout.addWidget(image_button)

        self.child_layout.addWidget(QLabel("Original Image"))
        self.child_layout.addWidget(QLabel("New Image"))

        # set the whole layout as the central widget
        self.main_layout.addLayout(self.child_layout)
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
            save_button.clicked.connect(self.show_download_dialog)
            save_button.setStatusTip("Click here to save your new fits images.")
            self.toolbar.addWidget(save_button)

    def show_download_dialog(self):
        # open the user's home directory
        dir = QFileDialog.getExistingDirectory(self, "Choose a Downnloads Folder", os.path.expanduser('~')) # self, browser name, default path can also be ""

        if dir:
            # If the user selected a path, proceed with download
            for path in processed_images:
                self.download_image(path, dir)
        else:
            print("Download cancelled by user.")

    def download_image(self, fits_path, destination_path):
        try:
            final_path = os.path.join(destination_path, os.path.basename(fits_path))

            shutil.copy2(fits_path, final_path)
            print(f"File downloaded successfully to: {destination_path}")
        except Exception as e:
            print(f"Error downloading file: {e}")

    def show_new_window(self):
        dialog = LoadImageWindow(parent=self)
        dialog.exec_()
        self.show_new_toolbar_main()

    def show_new_toolbar_main(self):
        global current_state
        if current_state == "Main_Window":
            # change state
            self.image_processing_state()
            current_state = "Image_Processing"

            # pop up loading screen after uploading images
            #self.show_loading_screen()

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

        self.loading_screen.start(self.client, list(self.image_data.keys()), self)
        # update the progress bar, and close 0.5s after getting to 100%
        #self.loading_screen.progress()
        #QTimer.singleShot(500, self.loading_screen.close)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.setGeometry(100, 100, 1300, 750)
    window.show()
    sys.exit(app.exec())