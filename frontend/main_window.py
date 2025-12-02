import os
import sys
from api_client import create_api_client

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
    def __init__(self, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).parent / "loading_screen.ui"
        loadUi(str(ui_path), self)

        self.progressBar.setValue(0)
        self.jobs_to_check = []
        self.client = None
        self.main_window = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_jobs)

    def start(self, client, job_ids, main_window):
        self.client = client
        self.jobs_to_check = job_ids
        self.main_window = main_window
        self.progressBar.setValue(0)
        self.show()
        self.timer.start(100) # check 10 times a second

    def check_jobs(self):
        try:
            completed_jobs = 0

            for job_id in self.jobs_to_check:
                try:
                    status = self.client.get_job_status(job_id)
                except Exception as e:
                    print(f"API error getting status for job {job_id}: {e}")
                    continue

                backend_status = status.get("status")
                print(f"Job {job_id}: status='{backend_status}'")

                # if images are still awaiting review, force corrections
                if backend_status == "AWAITING_REVIEW":
                    try:
                        detections = self.client.get_detections(job_id)
                    except Exception as e:
                        print(f"Detection fetch failed for job {job_id}: {e}")
                        detections = None

                    if detections:
                        # Store the trail count in image_data
                        self.main_window.image_data[job_id]["trail_count"] = detections.get("trail_count", 0)
                        trail_ids = [trail["trail_id"] for trail in detections["trails"]]
                        self.client.submit_corrections(job_id, trail_ids)

                if backend_status == "DONE":
                    completed_jobs += 1

                # compute average progress safely
                try:
                    progress = int((completed_jobs / len(self.jobs_to_check)) * 100) if self.jobs_to_check else 0
                except ZeroDivisionError:
                    print("Warning: No images uploaded to process.")
                    progress = 0

                self.progressBar.setValue(progress)
                QApplication.processEvents()

            # once all jobs are completed, stop timer and close
            if completed_jobs == len(self.jobs_to_check):
                self.timer.stop()
                self.close()
                print("All jobs completed!")
                self.main_window.image_processing_state()
        except Exception as e:
            print(f"Unexpected error in loading screen: {e}")


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
        
        # store all jobs
        job_ids = []

        for image in fits_images:
            image_path = Path(image)
            # get job id & status from backend
            job_id = self.parent_window.client.upload_image(image_path)

            # Skip if upload failed
            if job_id is None:
                print(f"Failed to upload {image_path}, skipping...")
                continue

            status = self.parent_window.client.get_job_status(job_id)
            self.parent_window.image_data[job_id] = {
                "original_path": image_path,
                "status": status
            }
            #add the job to the list
            job_ids.append(job_id)

        # Only start loading screen if we have jobs
        if job_ids:
            self.parent_window.loading_screen.start(self.parent_window.client, job_ids, self.parent_window)

        self.close()

        
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
                processed_png_path = Path(f"/tmp/{job_id}_processed.png")
                
                # convert from fits to png
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
        #global fits_images
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
            save_button.setStatusTip("Click here to save your new fits images.")
            self.toolbar.addWidget(save_button)

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