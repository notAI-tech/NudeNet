import qdarktheme
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QSlider,
    QListWidget,
    QFileDialog,
    QSplitter,
)


from liteindex import KVIndex, DefinedIndex

from SettingsBar import SettingsBar
from GalleryView import ImageGallery


images_index = DefinedIndex(
    "images_index",
    schema={
        "image_path": "string",
        "image_size": "number",
        "image_hash": "string",
        "FEMALE_GENITALIA_COVERED": "number",
        "FACE_FEMALE": "number",
        "BUTTOCKS_EXPOSED": "number",
        "FEMALE_BREAST_EXPOSED": "number",
        "FEMALE_GENITALIA_EXPOSED": "number",
        "MALE_BREAST_EXPOSED": "number",
        "ANUS_EXPOSED": "number",
        "FEET_EXPOSED": "number",
        "BELLY_COVERED": "number",
        "FEET_COVERED": "number",
        "ARMPITS_COVERED": "number",
        "ARMPITS_EXPOSED": "number",
        "FACE_MALE": "number",
        "BELLY_EXPOSED": "number",
        "MALE_GENITALIA_EXPOSED": "number",
        "ANUS_COVERED": "number",
        "FEMALE_BREAST_COVERED": "number",
        "BUTTOCKS_COVERED": "number",
    },
    db_path="images_index.db",
)


class NSFWSniffer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NSFW Sniffer")
        self.settings_index = KVIndex("settings.db")
        self.images_index = images_index

        splitter = QSplitter()

        self.settings_panel = SettingsBar(self.settings_index, self.images_index)
        splitter.addWidget(self.settings_panel)

        splitter.addWidget(ImageGallery(self.images_index))

        splitter.setSizes([300, 700])

        central_widget = QWidget()
        central_widget_layout = QHBoxLayout(central_widget)
        central_widget_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")

    window = NSFWSniffer()
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())
