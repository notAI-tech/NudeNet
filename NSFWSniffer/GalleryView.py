from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QScrollArea,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import sys


class ImageGallery(QMainWindow):
    def __init__(self, images_index, parent=None):
        super().__init__(parent)

        self.images_index = images_index

        # Initialize the scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # Initialize the central widget for the scroll area
        self.central_widget = QWidget()
        self.scroll_area.setWidget(self.central_widget)

        # Set the scroll area as the central widget of the main window
        self.setCentralWidget(self.scroll_area)

        # Set layout for the central widget
        self.layout = QVBoxLayout(self.central_widget)

        # Add a combo box for selecting column count
        self.column_selector = QComboBox(self)
        self.column_selector.addItems(["1", "2", "3", "4", "5"])
        self.column_selector.currentIndexChanged.connect(self.update_columns)
        self.layout.addWidget(self.column_selector)

        # Create the grid layout to hold the images
        self.grid_layout = QGridLayout()

        # Add the grid layout to the outer layout
        self.layout.addLayout(self.grid_layout)

        # Display initial images
        self.columns = 2
        self.display_images()

    def get_images_to_display(self, n=10, page_no=1):
        return [_["image_path"] for _ in self.images_index.search(query={}).values()]

    def update_columns(self):
        self.columns = int(self.column_selector.currentText())
        self.display_images()

    def display_images(self):
        # Clear out any existing widgets in the layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                self.grid_layout.removeWidget(widget)
                widget.deleteLater()

        # Get the images to display
        images = self.get_images_to_display()

        # Add images and checkboxes to the layout
        for i, image_path in enumerate(images):
            row = i // self.columns
            column = i % self.columns

            # Create a label and set a dummy pixmap (replace with actual image paths)
            label = QLabel()
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap.scaledToWidth(200, Qt.SmoothTransformation))

            # Create a checkbox for the image
            checkbox = QCheckBox("Image " + str(i + 1))

            self.grid_layout.addWidget(label, row, column * 2)
            self.grid_layout.addWidget(checkbox, row, column * 2 + 1)

        # Adjust spacing
        self.grid_layout.setHorizontalSpacing(10)
        self.grid_layout.setVerticalSpacing(10)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gallery = ImageGallery()
    gallery.show()
    sys.exit(app.exec_())
