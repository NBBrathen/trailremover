import sys
from PyQt5.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)

# create the window
window = QWidget()
window.setWindowTitle("My First PyQt App")
window.setGeometry(100, 100, 400, 300) # x, y, width, height

# show the window
window.show()

# start the application's event loop
sys.exit(app.exec_())
