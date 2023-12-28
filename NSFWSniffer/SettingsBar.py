# SettingsPanel.py

from PySide6.QtWidgets import (
    QVBoxLayout, QPushButton, QListWidget, QLineEdit, QSpinBox, QHBoxLayout, QLabel, QWidget, QFileDialog
)

class SettingsBar(QWidget):
    def __init__(self, settings_index, parent=None):
        super().__init__(parent)
        self.settings_index = settings_index
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
        self.folders_to_scan_list_widget.addItems(self.settings_index.get("folders_to_scan", ["No folders to scan"]))
        self.set_max_visible_items(self.folders_to_scan_list_widget, 5)
        sidebar_layout.addWidget(self.folders_to_scan_list_widget)

        self.folders_to_ignore_list_widget = QListWidget()
        self.folders_to_ignore_list_widget.addItems(self.settings_index.get("folders_to_ignore", ["No ignored folders"]))
        self.set_max_visible_items(self.folders_to_ignore_list_widget, 5)
        sidebar_layout.addWidget(self.folders_to_ignore_list_widget)

        # Minimum image resolution
        self.min_image_res = QLineEdit(self.settings_index.get("min_image_res", "48x48"))
        self.min_image_res.editingFinished.connect(self.save_min_image_res)
        min_image_res_widget = self.create_setting_widget("ignore less than resolution:", self.min_image_res)
        sidebar_layout.addWidget(min_image_res_widget)

        # Minimum file size
        self.min_file_size = QSpinBox()
        self.min_file_size.setMinimum(0)
        self.min_file_size.setValue(self.settings_index.get("min_file_size", 10))
        self.min_file_size.valueChanged.connect(self.save_min_file_size)
        min_file_size_widget = self.create_setting_widget("ignore less than KB:", self.min_file_size)
        sidebar_layout.addWidget(min_file_size_widget)

        return sidebar_layout

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
            self.settings_index["folders_to_scan"] = self.settings_index.get("folders_to_scan", []) + [folder_path]
            self.folders_to_scan_list_widget.addItem(folder_path)

    def ignore_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Ignore")
        if folder_path:
            self.settings_index["folders_to_ignore"] = self.settings_index.get("folders_to_ignore", []) + [folder_path]
            self.folders_to_ignore_list_widget.addItem(folder_path)

    def save_min_image_res(self):
        self.settings_index["min_image_res"] = self.min_image_res.text()

    def save_min_file_size(self):
        self.settings_index["min_file_size"] = self.min_file_size.value()
    
    def set_max_visible_items(self, list_widget, max_visible_items):
        item_height = list_widget.sizeHintForRow(0)
        total_height = item_height * max_visible_items
        list_widget.setFixedHeight(total_height)
