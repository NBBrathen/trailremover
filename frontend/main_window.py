import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

app = QApplication(sys.argv)
#app = QApplication([])

# create the window
window = QWidget()
window.setWindowTitle("TrailRemover")
window.setGeometry(100, 100, 1300, 750) # x, y, width, height
introMsg = QLabel("<h1>Welcome to the Trail Remover application!</h1>", parent=window)
introMsg.move(60, 15)

# show the window
window.show()

# run the application's main loop
sys.exit(app.exec_())
