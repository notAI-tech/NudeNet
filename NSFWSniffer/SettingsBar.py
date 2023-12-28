# SettingsPanel.py

from PySide6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QLineEdit,
    QSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
    QFileDialog,
)

from PySide6.QtCore import QTimer

from scanning_process import scanning_process
import threading


class SettingsBar(QWidget):
    def __init__(self, settings_index, images_index, parent=None):
        super().__init__(parent)
        self.settings_index = settings_index
        self.images_index = images_index
        self.setLayout(self.create_layout())

    def create_layout(self):
        sidebar_layout = QVBoxLayout(self)

        # Add folder input
        self.add_folder_btn = QPushButton("Add Folder for Scan")
        self.add_folder_btn.clicked.connect(self.add_folder)
        sidebar_layout.addWidget(self.add_folder_btn)

        # Ignore folder input
        self.ignore_folder_btn = QPushButton("Ignore Folder in Scan")
        self.ignore_folder_btn.clicked.connect(self.ignore_folder)
        sidebar_layout.addWidget(self.ignore_folder_btn)

        # Add ListWidgets to display folders
        self.folders_to_scan_list_widget = QListWidget()
        self.folders_to_scan_list_widget.addItems(
            self.settings_index.get("folders_to_scan", ["No folders to scan"])
        )
        self.set_max_visible_items(self.folders_to_scan_list_widget, 5)
        sidebar_layout.addWidget(self.folders_to_scan_list_widget)

        self.folders_to_ignore_list_widget = QListWidget()
        self.folders_to_ignore_list_widget.addItems(
            self.settings_index.get("folders_to_ignore", ["No ignored folders"])
        )
        self.set_max_visible_items(self.folders_to_ignore_list_widget, 5)
        sidebar_layout.addWidget(self.folders_to_ignore_list_widget)

        # Minimum image resolution
        self.min_image_res = QLineEdit(
            self.settings_index.get("min_image_res", "48x48")
        )
        self.min_image_res.editingFinished.connect(self.save_min_image_res)
        min_image_res_widget = self.create_setting_widget(
            "ignore less than resolution:", self.min_image_res
        )
        sidebar_layout.addWidget(min_image_res_widget)

        # Minimum file size
        self.min_file_size = QSpinBox()
        self.min_file_size.setMinimum(0)
        self.min_file_size.setValue(self.settings_index.get("min_file_size", 10))
        self.min_file_size.valueChanged.connect(self.save_min_file_size)
        min_file_size_widget = self.create_setting_widget(
            "ignore less than KB:", self.min_file_size
        )
        sidebar_layout.addWidget(min_file_size_widget)

        # Start scan button
        self.start_scan_btn = QPushButton("Start Scan")
        self.start_scan_btn.clicked.connect(self.start_scan)
        sidebar_layout.addWidget(self.start_scan_btn)

        self.scanner_thread = None

        # Number of items scanned and time elapsed
        self.scanned_items = QLabel(
            f"{self.get_number_of_scanned_items()} images scanned"
        )
        sidebar_layout.addWidget(self.scanned_items)

        # QTimer to update number of scanned items
        self.scanned_items_timer = QTimer()
        self.scanned_items_timer.timeout.connect(
            lambda: self.scanned_items.setText(
                f"{self.get_number_of_scanned_items()} images scanned"
            )
        )
        self.scanned_items_timer.start(1000)

        return sidebar_layout

    def get_number_of_scanned_items(self):
        return self.images_index.count()

    def start_scan(self):
        if not self.scanner_thread or not self.scanner_thread.is_alive():
            self.start_scan_btn.setEnabled(False)
            self.start_scan_btn.setText("Scanning...")
            self.start_scan_btn.repaint()
            self.scanner_thread = threading.Thread(
                target=scanning_process,
                args=(
                    self.settings_index["folders_to_scan"],
                    self.settings_index.get("folders_to_ignore", []),
                ),
            ).start()

    def create_setting_widget(self, label_text, setting_widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(setting_widget)
        container_widget = QWidget()
        container_widget.setLayout(layout)
        return container_widget

    def add_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder for Scan")
        if folder_path:
            folders_to_scan = list(
                set(self.settings_index.get("folders_to_scan", []) + [folder_path])
            )
            self.settings_index["folders_to_scan"] = folders_to_scan
            self.folders_to_scan_list_widget.clear()
            self.folders_to_scan_list_widget.addItem(folder_path)

    def ignore_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Ignore")
        if folder_path:
            folders_to_ignore = list(
                set(self.settings_index.get("folders_to_ignore", []) + [folder_path])
            )
            self.settings_index["folders_to_ignore"] = folders_to_ignore
            self.folders_to_ignore_list_widget.clear()
            self.folders_to_ignore_list_widget.addItem(folder_path)

    def save_min_image_res(self):
        self.settings_index["min_image_res"] = self.min_image_res.text()

    def save_min_file_size(self):
        self.settings_index["min_file_size"] = self.min_file_size.value()

    def set_max_visible_items(self, list_widget, max_visible_items):
        item_height = list_widget.sizeHintForRow(0)
        total_height = item_height * max_visible_items
        list_widget.setFixedHeight(total_height)
