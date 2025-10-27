import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QPushButton,
    QGridLayout
)

app = QApplication(sys.argv)

window = QWidget() # create the window
window.setWindowTitle("TrailRemover")
window.setGeometry(100, 100, 1300, 750)

layout = QGridLayout()
layout.addWidget( QLabel("<h1>Welcome to the Trail Remover application!</h1>"), 0, 0, 1, 3)
# TODO: move the label to the middle of the screen
#introMsg.move(375, 15)

layout.addWidget(QPushButton("Exit"), 1, 0)
layout.addWidget(QPushButton("Load Image"), 1, 2)
# TODO: move the buttons to the left and right
#exitButton.move(50, 400)

window.setLayout(layout)

# show the window
window.show()

# run the application's main loop
sys.exit(app.exec_())
