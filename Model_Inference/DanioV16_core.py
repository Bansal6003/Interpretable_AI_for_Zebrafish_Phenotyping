# DanioV16_core.py
# Red line (length): Extends exactly along the longest dimension of the mask
# Blue line (width): Extends exactly along the shortest dimension of the mask
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import json
import torch
from torch import nn
from torchvision import transforms
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QMessageBox, QApplication, QProgressDialog,
                           QDialog, QLineEdit, QFormLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import timm
import segmentation_models_pytorch as smp
from PIL import Image
from ultralytics import YOLO
import pandas as pd

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QApplication, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights
# from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights

LARGE_BATCH_THRESHOLD = 800  # Consider "large batch" if more than this
BATCH_SIZE_LARGE = 30        # Smaller batches for large datasets
BATCH_SIZE_NORMAL = 50       # Normal batch size
MAX_DISPLAY_IMAGES_LARGE = 50   # Fewer images kept for display in large batches
MAX_DISPLAY_IMAGES_NORMAL = 100 # Normal display limit

# Memory thresholds
MEMORY_WARNING_THRESHOLD = 6000  # MB - warn when memory usage exceeds this
MEMORY_CRITICAL_THRESHOLD = 8000 # MB - force aggressive cleanup

def _natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Danio Analyzer Platform")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)

        # Title
        title = QLabel("Danio Analyzer Platform")
        title.setFont(QFont('Arial', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Buttons
        class_seg_btn = QPushButton("Classification and Segmentation")
        class_seg_btn.setFont(QFont('Arial', 14))
        class_seg_btn.setFixedSize(400, 60)
        class_seg_btn.clicked.connect(self.open_classification)
        layout.addWidget(class_seg_btn, alignment=Qt.AlignCenter)

        behavior_btn = QPushButton("Behavior Analysis")
        behavior_btn.setFont(QFont('Arial', 14))
        behavior_btn.setFixedSize(400, 60)
        behavior_btn.clicked.connect(self.open_behavior)
        layout.addWidget(behavior_btn, alignment=Qt.AlignCenter)

    def open_classification(self):
        self.analyzer_window = DanioAnalyzerGUI()
        self.analyzer_window.show()

    def open_behavior(self):
        # Placeholder for future behavior analysis implementation
        pass

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
EFFICIENTNET_PATH = r".\Best_Models_regularly_updated\Class_seg_models\Efficient_NetV2_s_WT_vs_Mutant_DarkBG_p20.pt"
# EFFICIENTNET_PATH = r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\V36_same as V35_with_added_orientation_class_model\V36_same as V35_with_added_orientation_class_model\CODE FILES\Best_Models_regularly_updated\Class_seg_models\Efficient_NetV2_auged_mic_drop_meet_V3_21K.pt"
DEEPLABV3_PATH = r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_whole_larva.pth"
YOLO_PATH = r".\Best_Models_regularly_updated\Class_seg_models\YOLO_Whole_Larva_last.pt"
CLASS_NAMES = ['WT', 'mutant']
# CLASS_NAMES = ['Dorsal_all', 'Lateral_all']

ORIENTATION_MODEL_PATH = r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\RegNet_Y_16_Pretrained\RegNet_Y_16GF_2_class_orients_cropped_P50_2.pt"
# ORIENTATION_MODEL_PATH = r'.\Best_Models_regularly_updated\Class_seg_models\RegNet_Y_16GF_Orients_Dark_BG_p50_2.pt'
# ORIENTATION_MODEL_PATH = r'D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\RegNet_Y_16_Pretrained\RegNet_Y_32GF_Orients_Dark_BG_p75_1.pt'

# ORIENTATION_CLASS_NAMES = ['Lateral', 'Ventral', 'Ventro Lateral']
ORIENTATION_CLASS_NAMES = [ 'Ventral' , 'latera_VentroLateral']

# Transform for Orientation Classification (if different from phenotype model)
orientation_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class OrientationEfficientNetV2(nn.Module):
#     def __init__(self, num_classes=3):  # 3 orientations
#         super().__init__()
#         self.efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True)
#         num_features = self.efficientnet.classifier.in_features
#         self.efficientnet.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(num_features, num_classes)
#         )

#     def forward(self, x):
#         return self.efficientnet(x)

class OrientationEfficientNetV2(nn.Module): #this is actually RegNetY_16GF
    def __init__(self, num_classes=2):
        super(OrientationEfficientNetV2, self).__init__()
        # Load RegNet_Y_16GF with IMAGENET1K_SWAG_E2E_V1 weights
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        self.regnet = regnet_y_16gf(weights=weights)
        
        # Modify the classifier head
        num_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.regnet(x)

class ControlValueDialog(QDialog):
    def __init__(self, analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.setWindowTitle("Set Control Value Ranges")
        self.setFixedSize(800, 600)  # Make dialog larger to accommodate more fields
        self.setup_ui()
        
    
    # Modified ControlValueDialog.setup_ui() method - with dual controls for pericardium
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("Set Control Value Ranges for Measurements")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create scroll area for the form
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.value_inputs = {}
        # orientations = ["Ventral", "Lateral", "Ventro Lateral"]
        orientations = ['Ventral' , 'latera_VentroLateral']
        
        # Use segmentation model names instead of class names
        for model_name in self.analyzer.segmentation_models.keys():
            # Create group box for each segmentation model
            from PyQt5.QtWidgets import QGroupBox
            model_group = QGroupBox(model_name)
            model_group.setFont(QFont('Arial', 12, QFont.Bold))
            model_layout = QFormLayout(model_group)
            
            self.value_inputs[model_name] = {}
            
            for orientation in orientations:
                # Create sub-group for each orientation
                orientation_group = QGroupBox(orientation)
                orientation_layout = QFormLayout(orientation_group)
                
                
                if model_name == 'Pericardium':
                    # Pericardium has area, box length, and box width controls
                    # Area controls
                    min_area_edit = QLineEdit()
                    min_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_area', 0.0)
                    min_area_edit.setText(str(current_min_area))
                    orientation_layout.addRow("Min Area (mm²):", min_area_edit)
                    
                    max_area_edit = QLineEdit()
                    max_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_area', 0.0)
                    max_area_edit.setText(str(current_max_area))
                    orientation_layout.addRow("Max Area (mm²):", max_area_edit)
                    
                    # Area threshold
                    area_threshold_edit = QLineEdit()
                    area_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_area_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('area_threshold', 5.0)
                    area_threshold_edit.setText(str(current_area_threshold))
                    orientation_layout.addRow("Area Threshold (%):", area_threshold_edit)
                    
                    # Box Length controls
                    min_length_edit = QLineEdit()
                    min_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_length', 0.0)
                    min_length_edit.setText(str(current_min_length))
                    orientation_layout.addRow("Min Box Length (mm):", min_length_edit)
                    
                    max_length_edit = QLineEdit()
                    max_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_length', 0.0)
                    max_length_edit.setText(str(current_max_length))
                    orientation_layout.addRow("Max Box Length (mm):", max_length_edit)
                    
                    # Length threshold
                    length_threshold_edit = QLineEdit()
                    length_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_length_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('length_threshold', 5.0)
                    length_threshold_edit.setText(str(current_length_threshold))
                    orientation_layout.addRow("Length Threshold (%):", length_threshold_edit)
                    
                    # Box Width controls
                    min_width_edit = QLineEdit()
                    min_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_width', 0.0)
                    min_width_edit.setText(str(current_min_width))
                    orientation_layout.addRow("Min Box Width (mm):", min_width_edit)
                    
                    max_width_edit = QLineEdit()
                    max_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_width', 0.0)
                    max_width_edit.setText(str(current_max_width))
                    orientation_layout.addRow("Max Box Width (mm):", max_width_edit)
                    
                    # Width threshold
                    width_threshold_edit = QLineEdit()
                    width_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_width_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('width_threshold', 5.0)
                    width_threshold_edit.setText(str(current_width_threshold))
                    orientation_layout.addRow("Width Threshold (%):", width_threshold_edit)
                    
                    # Store inputs for pericardium
                    self.value_inputs[model_name][orientation] = {
                        'min_area': min_area_edit,
                        'max_area': max_area_edit,
                        'area_threshold': area_threshold_edit,
                        'min_length': min_length_edit,
                        'max_length': max_length_edit,
                        'length_threshold': length_threshold_edit,
                        'min_width': min_width_edit,
                        'max_width': max_width_edit,
                        'width_threshold': width_threshold_edit
                    }
                    
                elif model_name == 'Yolksac_ext':
                    # Yolksac_ext has area, box length, and box width controls with separate thresholds
                    # Area controls
                    min_area_edit = QLineEdit()
                    min_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_area', 0.0)
                    min_area_edit.setText(str(current_min_area))
                    orientation_layout.addRow("Min Area (mm²):", min_area_edit)
                    
                    max_area_edit = QLineEdit()
                    max_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_area', 0.0)
                    max_area_edit.setText(str(current_max_area))
                    orientation_layout.addRow("Max Area (mm²):", max_area_edit)
                    
                    # Area threshold
                    area_threshold_edit = QLineEdit()
                    area_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_area_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('area_threshold', 5.0)
                    area_threshold_edit.setText(str(current_area_threshold))
                    orientation_layout.addRow("Area Threshold (%):", area_threshold_edit)
                    
                    # Box Length controls
                    min_length_edit = QLineEdit()
                    min_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_length', 0.0)
                    min_length_edit.setText(str(current_min_length))
                    orientation_layout.addRow("Min Box Length (mm):", min_length_edit)
                    
                    max_length_edit = QLineEdit()
                    max_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_length', 0.0)
                    max_length_edit.setText(str(current_max_length))
                    orientation_layout.addRow("Max Box Length (mm):", max_length_edit)
                    
                    # Length threshold
                    length_threshold_edit = QLineEdit()
                    length_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_length_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('length_threshold', 5.0)
                    length_threshold_edit.setText(str(current_length_threshold))
                    orientation_layout.addRow("Length Threshold (%):", length_threshold_edit)
                    
                    # Box Width controls
                    min_width_edit = QLineEdit()
                    min_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_width', 0.0)
                    min_width_edit.setText(str(current_min_width))
                    orientation_layout.addRow("Min Box Width (mm):", min_width_edit)
                    
                    max_width_edit = QLineEdit()
                    max_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_width', 0.0)
                    max_width_edit.setText(str(current_max_width))
                    orientation_layout.addRow("Max Box Width (mm):", max_width_edit)
                    
                    # Width threshold
                    width_threshold_edit = QLineEdit()
                    width_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_width_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('width_threshold', 5.0)
                    width_threshold_edit.setText(str(current_width_threshold))
                    orientation_layout.addRow("Width Threshold (%):", width_threshold_edit)
                    
                    # Store inputs for yolksac_ext
                    self.value_inputs[model_name][orientation] = {
                        'min_area': min_area_edit,
                        'max_area': max_area_edit,
                        'area_threshold': area_threshold_edit,
                        'min_length': min_length_edit,
                        'max_length': max_length_edit,
                        'length_threshold': length_threshold_edit,
                        'min_width': min_width_edit,
                        'max_width': max_width_edit,
                        'width_threshold': width_threshold_edit
                    }
                
                elif model_name == 'Head':
                    # Head has area, length, and width controls
                    # Area controls
                    min_area_edit = QLineEdit()
                    min_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_area', 0.0)
                    min_area_edit.setText(str(current_min_area))
                    orientation_layout.addRow("Min Area (mm²):", min_area_edit)
                    
                    max_area_edit = QLineEdit()
                    max_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_area', 0.0)
                    max_area_edit.setText(str(current_max_area))
                    orientation_layout.addRow("Max Area (mm²):", max_area_edit)
                    
                    # Area threshold
                    area_threshold_edit = QLineEdit()
                    area_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_area_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('area_threshold', 5.0)
                    area_threshold_edit.setText(str(current_area_threshold))
                    orientation_layout.addRow("Area Threshold (%):", area_threshold_edit)
                    
                    # Length controls
                    min_length_edit = QLineEdit()
                    min_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_length', 0.0)
                    min_length_edit.setText(str(current_min_length))
                    orientation_layout.addRow("Min Length (mm):", min_length_edit)
                    
                    max_length_edit = QLineEdit()
                    max_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_length', 0.0)
                    max_length_edit.setText(str(current_max_length))
                    orientation_layout.addRow("Max Length (mm):", max_length_edit)
                    
                    # Length threshold
                    length_threshold_edit = QLineEdit()
                    length_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_length_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('length_threshold', 5.0)
                    length_threshold_edit.setText(str(current_length_threshold))
                    orientation_layout.addRow("Length Threshold (%):", length_threshold_edit)
                    
                    # Width controls
                    min_width_edit = QLineEdit()
                    min_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_width', 0.0)
                    min_width_edit.setText(str(current_min_width))
                    orientation_layout.addRow("Min Width (mm):", min_width_edit)
                    
                    max_width_edit = QLineEdit()
                    max_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_width', 0.0)
                    max_width_edit.setText(str(current_max_width))
                    orientation_layout.addRow("Max Width (mm):", max_width_edit)
                    
                    # Width threshold
                    width_threshold_edit = QLineEdit()
                    width_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_width_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('width_threshold', 5.0)
                    width_threshold_edit.setText(str(current_width_threshold))
                    orientation_layout.addRow("Width Threshold (%):", width_threshold_edit)
                    
                    # Store inputs for head
                    self.value_inputs[model_name][orientation] = {
                        'min_area': min_area_edit,
                        'max_area': max_area_edit,
                        'area_threshold': area_threshold_edit,
                        'min_length': min_length_edit,
                        'max_length': max_length_edit,
                        'length_threshold': length_threshold_edit,
                        'min_width': min_width_edit,
                        'max_width': max_width_edit,
                        'width_threshold': width_threshold_edit
                    }
                    
                elif model_name == 'Yolk_Sac':
                    min_area_edit = QLineEdit()
                    min_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_area', 0.0)
                    min_area_edit.setText(str(current_min_area))
                    orientation_layout.addRow("Min Area (mm²):", min_area_edit)
                    
                    max_area_edit = QLineEdit()
                    max_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_area', 0.0)
                    max_area_edit.setText(str(current_max_area))
                    orientation_layout.addRow("Max Area (mm²):", max_area_edit)
                    
                    # Area threshold
                    area_threshold_edit = QLineEdit()
                    area_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_area_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('area_threshold', 5.0)
                    area_threshold_edit.setText(str(current_area_threshold))
                    orientation_layout.addRow("Area Threshold (%):", area_threshold_edit)
                    
                    # Box Length controls
                    min_length_edit = QLineEdit()
                    min_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_length', 0.0)
                    min_length_edit.setText(str(current_min_length))
                    orientation_layout.addRow("Min Box Length (mm):", min_length_edit)
                    
                    max_length_edit = QLineEdit()
                    max_length_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_length = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_length', 0.0)
                    max_length_edit.setText(str(current_max_length))
                    orientation_layout.addRow("Max Box Length (mm):", max_length_edit)
                    
                    # Length threshold
                    length_threshold_edit = QLineEdit()
                    length_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_length_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('length_threshold', 5.0)
                    length_threshold_edit.setText(str(current_length_threshold))
                    orientation_layout.addRow("Length Threshold (%):", length_threshold_edit)
                    
                    # Box Width controls
                    min_width_edit = QLineEdit()
                    min_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_width', 0.0)
                    min_width_edit.setText(str(current_min_width))
                    orientation_layout.addRow("Min Box Width (mm):", min_width_edit)
                    
                    max_width_edit = QLineEdit()
                    max_width_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_width = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_width', 0.0)
                    max_width_edit.setText(str(current_max_width))
                    orientation_layout.addRow("Max Box Width (mm):", max_width_edit)
                    
                    # Width threshold
                    width_threshold_edit = QLineEdit()
                    width_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_width_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('width_threshold', 5.0)
                    width_threshold_edit.setText(str(current_width_threshold))
                    orientation_layout.addRow("Width Threshold (%):", width_threshold_edit)
                    
                    # Store inputs for pericardium
                    self.value_inputs[model_name][orientation] = {
                        'min_area': min_area_edit,
                        'max_area': max_area_edit,
                        'area_threshold': area_threshold_edit,
                        'min_length': min_length_edit,
                        'max_length': max_length_edit,
                        'length_threshold': length_threshold_edit,
                        'min_width': min_width_edit,
                        'max_width': max_width_edit,
                        'width_threshold': width_threshold_edit
                    }
                    
                else:
                    # All other models use area AND box length with separate thresholds
                    # Area controls
                    min_area_edit = QLineEdit()
                    min_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min_area', 0.0)
                    min_area_edit.setText(str(current_min_area))
                    orientation_layout.addRow("Min Area (mm²):", min_area_edit)
                    
                    max_area_edit = QLineEdit()
                    max_area_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max_area = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max_area', 0.0)
                    max_area_edit.setText(str(current_max_area))
                    orientation_layout.addRow("Max Area (mm²):", max_area_edit)
                    
                    # Area threshold
                    area_threshold_edit = QLineEdit()
                    area_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_area_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('area_threshold', 5.0)
                    area_threshold_edit.setText(str(current_area_threshold))
                    orientation_layout.addRow("Area Threshold (%):", area_threshold_edit)
                    
                    # Box Length controls
                    min_edit = QLineEdit()
                    min_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_min = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('min', 0.0)
                    min_edit.setText(str(current_min))
                    orientation_layout.addRow("Min Box Length (mm):", min_edit)
                    
                    max_edit = QLineEdit()
                    max_edit.setValidator(QDoubleValidator(0.0, 9999.99, 3))
                    current_max = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('max', 0.0)
                    max_edit.setText(str(current_max))
                    orientation_layout.addRow("Max Box Length (mm):", max_edit)
                    
                    # Length threshold
                    length_threshold_edit = QLineEdit()
                    length_threshold_edit.setValidator(QDoubleValidator(0.0, 50.0, 1))
                    current_length_threshold = self.analyzer.control_values.get(model_name, {}).get(orientation, {}).get('length_threshold', 5.0)
                    length_threshold_edit.setText(str(current_length_threshold))
                    orientation_layout.addRow("Length Threshold (%):", length_threshold_edit)
                    
                    # Store inputs for other models (now with area and separate thresholds)
                    self.value_inputs[model_name][orientation] = {
                        'min_area': min_area_edit,
                        'max_area': max_area_edit,
                        'area_threshold': area_threshold_edit,
                        'min': min_edit,
                        'max': max_edit,
                        'length_threshold': length_threshold_edit
                    }
                
                model_layout.addRow(orientation_group)
            
            scroll_layout.addWidget(model_group)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_values)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def save_values(self):
        try:
            for model_name, orientations in self.value_inputs.items():
                for orientation, inputs in orientations.items():
                    
                    if model_name == 'Pericardium':
                        # Handle pericardium with area, length, width and their thresholds
                        min_area = float(inputs['min_area'].text() or 0)
                        max_area = float(inputs['max_area'].text() or 0)
                        area_threshold = float(inputs['area_threshold'].text() or 5.0)
                        min_length = float(inputs['min_length'].text() or 0)
                        max_length = float(inputs['max_length'].text() or 0)
                        length_threshold = float(inputs['length_threshold'].text() or 5.0)
                        min_width = float(inputs['min_width'].text() or 0)
                        max_width = float(inputs['max_width'].text() or 0)
                        width_threshold = float(inputs['width_threshold'].text() or 5.0)
                        
                        # Validate that min <= max for all three dimensions
                        if min_area > max_area and max_area > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum area cannot be greater than maximum area")
                            return
                        if min_length > max_length and max_length > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum length cannot be greater than maximum length")
                            return
                        if min_width > max_width and max_width > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum width cannot be greater than maximum width")
                            return
                        
                        # Update control values for pericardium
                        if model_name not in self.analyzer.control_values:
                            self.analyzer.control_values[model_name] = {}
                        if orientation not in self.analyzer.control_values[model_name]:
                            self.analyzer.control_values[model_name][orientation] = {}
                        
                        self.analyzer.control_values[model_name][orientation] = {
                            'min_area': min_area, 'max_area': max_area, 'area_threshold': area_threshold,
                            'min_length': min_length, 'max_length': max_length, 'length_threshold': length_threshold,
                            'min_width': min_width, 'max_width': max_width, 'width_threshold': width_threshold
                        }
                    
                    elif model_name == 'Yolksac_ext':
                        # Handle yolksac_ext with area, box length, and box width and their thresholds
                        min_area = float(inputs['min_area'].text() or 0)
                        max_area = float(inputs['max_area'].text() or 0)
                        area_threshold = float(inputs['area_threshold'].text() or 5.0)
                        min_length = float(inputs['min_length'].text() or 0)
                        max_length = float(inputs['max_length'].text() or 0)
                        length_threshold = float(inputs['length_threshold'].text() or 5.0)
                        min_width = float(inputs['min_width'].text() or 0)
                        max_width = float(inputs['max_width'].text() or 0)
                        width_threshold = float(inputs['width_threshold'].text() or 5.0)
                        
                        # Validate that min <= max for all three dimensions
                        if min_area > max_area and max_area > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Yolksac_ext - {orientation}: Minimum area cannot be greater than maximum area")
                            return
                        if min_length > max_length and max_length > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Yolksac_ext - {orientation}: Minimum box length cannot be greater than maximum box length")
                            return
                        if min_width > max_width and max_width > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Yolksac_ext - {orientation}: Minimum box width cannot be greater than maximum box width")
                            return
                        
                        # Update control values for yolksac_ext
                        if model_name not in self.analyzer.control_values:
                            self.analyzer.control_values[model_name] = {}
                        if orientation not in self.analyzer.control_values[model_name]:
                            self.analyzer.control_values[model_name][orientation] = {}
                        
                        self.analyzer.control_values[model_name][orientation] = {
                            'min_area': min_area, 'max_area': max_area, 'area_threshold': area_threshold,
                            'min_length': min_length, 'max_length': max_length, 'length_threshold': length_threshold,
                            'min_width': min_width, 'max_width': max_width, 'width_threshold': width_threshold
                        }
                        
                    elif model_name == 'Head':
                        # Handle pericardium with area, length, width and their thresholds
                        min_area = float(inputs['min_area'].text() or 0)
                        max_area = float(inputs['max_area'].text() or 0)
                        area_threshold = float(inputs['area_threshold'].text() or 5.0)
                        min_length = float(inputs['min_length'].text() or 0)
                        max_length = float(inputs['max_length'].text() or 0)
                        length_threshold = float(inputs['length_threshold'].text() or 5.0)
                        min_width = float(inputs['min_width'].text() or 0)
                        max_width = float(inputs['max_width'].text() or 0)
                        width_threshold = float(inputs['width_threshold'].text() or 5.0)
                        
                        # Validate that min <= max for all three dimensions
                        if min_area > max_area and max_area > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum area cannot be greater than maximum area")
                            return
                        if min_length > max_length and max_length > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum length cannot be greater than maximum length")
                            return
                        if min_width > max_width and max_width > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum width cannot be greater than maximum width")
                            return
                        
                        # Update control values for pericardium
                        if model_name not in self.analyzer.control_values:
                            self.analyzer.control_values[model_name] = {}
                        if orientation not in self.analyzer.control_values[model_name]:
                            self.analyzer.control_values[model_name][orientation] = {}
                        
                        self.analyzer.control_values[model_name][orientation] = {
                            'min_area': min_area, 'max_area': max_area, 'area_threshold': area_threshold,
                            'min_length': min_length, 'max_length': max_length, 'length_threshold': length_threshold,
                            'min_width': min_width, 'max_width': max_width, 'width_threshold': width_threshold
                        }
                    
                    elif model_name == 'Yolk_Sac':
                        # Handle pericardium with area, length, width and their thresholds
                        min_area = float(inputs['min_area'].text() or 0)
                        max_area = float(inputs['max_area'].text() or 0)
                        area_threshold = float(inputs['area_threshold'].text() or 5.0)
                        min_length = float(inputs['min_length'].text() or 0)
                        max_length = float(inputs['max_length'].text() or 0)
                        length_threshold = float(inputs['length_threshold'].text() or 5.0)
                        min_width = float(inputs['min_width'].text() or 0)
                        max_width = float(inputs['max_width'].text() or 0)
                        width_threshold = float(inputs['width_threshold'].text() or 5.0)
                        
                        # Validate that min <= max for all three dimensions
                        if min_area > max_area and max_area > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum area cannot be greater than maximum area")
                            return
                        if min_length > max_length and max_length > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum length cannot be greater than maximum length")
                            return
                        if min_width > max_width and max_width > 0:
                            QMessageBox.warning(self, "Error", 
                                f"For Pericardium - {orientation}: Minimum width cannot be greater than maximum width")
                            return
                        
                        # Update control values for pericardium
                        if model_name not in self.analyzer.control_values:
                            self.analyzer.control_values[model_name] = {}
                        if orientation not in self.analyzer.control_values[model_name]:
                            self.analyzer.control_values[model_name][orientation] = {}
                        
                        self.analyzer.control_values[model_name][orientation] = {
                            'min_area': min_area, 'max_area': max_area, 'area_threshold': area_threshold,
                            'min_length': min_length, 'max_length': max_length, 'length_threshold': length_threshold,
                            'min_width': min_width, 'max_width': max_width, 'width_threshold': width_threshold
                        }
                        
                    else:
                        # Handle other models with area + box length and their thresholds
                        min_area = float(inputs['min_area'].text() or 0)
                        max_area = float(inputs['max_area'].text() or 0)
                        area_threshold = float(inputs['area_threshold'].text() or 5.0)
                        min_val = float(inputs['min'].text() or 0)
                        max_val = float(inputs['max'].text() or 0)
                        length_threshold = float(inputs['length_threshold'].text() or 5.0)
                        
                        # Validation and saving logic...
                        self.analyzer.control_values[model_name][orientation] = {
                            'min_area': min_area, 'max_area': max_area, 'area_threshold': area_threshold,
                            'min': min_val, 'max': max_val, 'length_threshold': length_threshold
                        }
            
            # Save the control values
            self.analyzer.save_control_values(self.analyzer.control_values)
            self.accept()
            QMessageBox.information(self, "Success", "Control value ranges updated successfully")
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        num_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

CLASS_NAMES = ['WT', 'mutant']
# CLASS_NAMES = ['Dorsal_all', 'Laterall_all']
# 
class ImageAnalyzer:
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.orientation_names = ORIENTATION_CLASS_NAMES
        self.pixel_size = 0.01
        
        # Define transforms as instance variables
        self.phenotype_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.orientation_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Segmentation models with encoder type specification
        # Format: 'name': {'path': 'model_path', 'encoder': 'resnet' or 'regnet'}
        self.segmentation_models = {
            'Whole Larva': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_Whole_larva_2_patience50.pth_2",
                'encoder': 'resnet'
            },
            'Ventral Eye': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_Ventral_eye_patience50_5.pth",
                'encoder': 'resnet'
            },
            'Ventro Lateral eye': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_ventro_lateral_eye_with_traf_patience50_2.pth",
                'encoder': 'resnet'
            },
            'Head': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_head_resized_patience50_2.pth",
                'encoder': 'resnet'
            },
            # 'Pericardium': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_pericardium_official_patience100_4.pth",
            'Pericardium': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\regnet_y_16_pericardium_patience100_1.pth",
                'encoder': 'regnet'
            },
            # 'Notochord': r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\V36_same as V35_with_added_orientation_class_model\V36_same as V35_with_added_orientation_class_model\CODE FILES\Best_Models_regularly_updated\Class_seg_models\Notochord_deeplabv3plus_final_model.pth",
            'Yolksac_ext': {
                # 'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_yolk_sac_ext_patience100_4.pth",
                'path' : r".\Best_Models_regularly_updated\Class_seg_models\regnet_y_16_yolk_sac_extension_patience75_1.pth",
                'encoder': 'regnet'
            },
            'Yolk_Sac': {
                'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_yolk_sac_with_traf_Li_patience50.pth",
                'encoder': 'resnet'
            },
            'Tail_extension': {
                # 'path': r".\Best_Models_regularly_updated\Class_seg_models\resnet50_swsl_echo_tail_ext_official_patience100_2.pth",
                'path': r".\Best_Models_regularly_updated\Class_seg_models\regnet_y_16_tail_extension_patience100_1.pth",
                'encoder': 'regnet'
            }
        }
        self.current_segmentation = 'Whole Larva'
        self.control_values = self.load_control_values()
        self.setup_models()
    
    def setup_models(self):
        # Load EfficientNet for phenotype classification
        self.classification_model = CustomEfficientNetV2(num_classes=len(self.class_names))
        self.classification_model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
        self.classification_model.eval()
        self.classification_model.to(device)
    
        # Load EfficientNet for orientation classification
        try:
            self.orientation_model = OrientationEfficientNetV2(num_classes=len(self.orientation_names))
            self.orientation_model.load_state_dict(torch.load(ORIENTATION_MODEL_PATH, map_location=device))
            self.orientation_model.eval()
            self.orientation_model.to(device)
            print("Orientation classification model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load orientation model: {str(e)}")
            print("Orientation classification will default to 'Ventral'")
            self.orientation_model = None
    
        # Load DeepLabV3+ models for segmentation
        self.segmentation_models_loaded = {}
        for name, model_info in self.segmentation_models.items():
            path = model_info['path']
            encoder_type = model_info['encoder']

            # Create model with appropriate encoder
            if encoder_type == 'resnet':
                model = smp.DeepLabV3Plus(
                    encoder_name="resnet50",
                    encoder_weights="swsl",
                    in_channels=3,
                    classes=1
                )
                print(f"Loading {name} with ResNet50 encoder...")
            elif encoder_type == 'regnet':
                model = smp.DeepLabV3Plus(
                    encoder_name="timm-regnety_160",
                    encoder_weights="swsl",
                    in_channels=3,
                    classes=1
                )
                print(f"Loading {name} with RegNet Y 160 encoder...")
            else:
                print(f"Warning: Unknown encoder type '{encoder_type}' for {name}, defaulting to ResNet50")
                model = smp.DeepLabV3Plus(
                    encoder_name="resnet50",
                    encoder_weights="swsl",
                    in_channels=3,
                    classes=1
                )

            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            model.to(device)
            self.segmentation_models_loaded[name] = model
            print(f"Successfully loaded {name} segmentation model")
    
        # Load YOLO for isolation
        self.yolo_model = YOLO(YOLO_PATH)
    
    def process_folder(self, folder_path):
        results = []
        processed_images = []
        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))],
            key=_natural_sort_key)

        for filename in image_files:
            try:
                image_path = os.path.join(folder_path, filename)
                
                # Process image
                isolated_image = self.isolate_object(image_path)
                if isolated_image is None:
                    continue
    
                mask, original_image, length, width, ratio, box_length, box_width, vis_mask = self.segment_image(image_path)
                area = self.calculate_area(mask)
                
                class_name, confidence = self.classify_image(isolated_image)
                status, percentage = self.check_anomaly(area, class_name)
                
                # Store results
                result = {
                    'Image': filename,
                    'Classification': class_name,
                    'Confidence': confidence,
                    'Area': area,
                    'Status': status,
                    'Percentage': percentage
                }
                results.append(result)
                
                # Store processed images
                processed_images.append({
                    'original': original_image,
                    'mask': mask,
                    'overlay': self.create_overlay(original_image, mask)
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
        return results, processed_images
    
    def classify_orientation(self, image):
        """Classify the orientation of the image"""
        if self.orientation_model is None:
            return [("Ventral", 1.0)]  # Default fallback
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.orientation_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.orientation_model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            
            # Get all orientations above threshold
            threshold = 0.1  # Lower threshold for orientation
            orientations = []
            for idx, prob in enumerate(probs):
                confidence = prob.item()
                if confidence > threshold:
                    orientations.append((self.orientation_names[idx], confidence))
            
            # Sort by confidence
            orientations.sort(key=lambda x: x[1], reverse=True)
            
            # Ensure at least one orientation is returned
            if not orientations:
                max_idx = torch.argmax(probs).item()
                orientations = [(self.orientation_names[max_idx], probs[max_idx].item())]
                
            return orientations

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        original_size = image.size
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        input_tensor = transform(image)
        input_tensor = input_tensor.repeat(3, 1, 1)
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor, image, original_size

    def isolate_object(self, image_path):
        try:
            results = self.yolo_model(image_path)
            if not results or not results[0].boxes:
                return None
            
            image = cv2.imread(image_path)
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            isolated_image = image[y1:y2, x1:x2]
            return isolated_image
        except Exception as e:
            print(f"Error in isolate_object: {str(e)}")
            return None

    def segment_image(self, image_path):
        try:
            input_tensor, original_image, original_size = self.preprocess_image(image_path)
            model = self.segmentation_models_loaded[self.current_segmentation]
            
            with torch.no_grad():
                output = model(input_tensor.to(device))
                mask = torch.sigmoid(output) > 0.5
                mask = mask.float().squeeze().cpu().numpy()
                
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize(original_size, Image.NEAREST)
                mask = np.array(mask_img) > 127
                
                # Calculate dimensions
                try:
                    length_mm, width_mm, ratio, box_length_mm, box_width_mm, vis_mask = self.calculate_dimensions(mask)
                except Exception as e:
                    print(f"Warning: Could not calculate dimensions for {image_path}: {str(e)}")
                    # Provide default values when calculation fails
                    length_mm, width_mm, ratio, box_length_mm, box_width_mm = 0.0, 0.0, 0.0, 0.0, 0.0
                    vis_mask = np.zeros_like(np.array(original_image))
                    if len(vis_mask.shape) == 2:
                        vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)

                # Compute and overlay curvature for Whole Larva model only
                self._last_curvature_label = 'N/A'
                if self.current_segmentation == 'Whole Larva':
                    try:
                        curv_label, curv_angle = self.calculate_curvature(mask)
                        self._last_curvature_label = curv_label
                        curv_color = (0, 200, 0) if curv_label == 'No Curvature' else (0, 0, 255)
                        text_y = max(vis_mask.shape[0] - 12, 20)
                        cv2.putText(vis_mask, curv_label, (10, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, curv_color, 1, cv2.LINE_AA)
                    except Exception:
                        self._last_curvature_label = 'N/A'

            return mask.astype(np.uint8), original_image, length_mm, width_mm, ratio, box_length_mm, box_width_mm, vis_mask
        
        except Exception as e:
            print(f"Error in segment_image for {image_path}: {str(e)}")
            # Return placeholder values when segmentation fails
            placeholder_image = Image.new('RGB', (100, 100), color='black')
            placeholder_mask = np.zeros((100, 100), dtype=np.uint8)
            placeholder_vis = np.zeros((100, 100, 3), dtype=np.uint8)
            return placeholder_mask, placeholder_image, 0.0, 0.0, 0.0, 0.0, 0.0, placeholder_vis

    def calculate_dimensions(self, mask):
        """Calculate length, width, ratio, box dimensions with edge-to-edge measurements"""
        import cv2
        import numpy as np
        
        # Convert mask to uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find all contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Create an empty visualization mask with 3 channels
            empty_vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            return 0.0, 0.0, 0.0, 0.0, 0.0, empty_vis_mask
        
        # Find the largest contour (main region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        clean_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Update the original mask to only include the largest region
        mask_uint8 = clean_mask
        
        # Find non-zero points (mask pixels) from the cleaned mask
        y_coords, x_coords = np.nonzero(mask_uint8)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            # Create an empty visualization mask with 3 channels
            empty_vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            return 0.0, 0.0, 0.0, 0.0, 0.0, empty_vis_mask
        
        # Create visualization mask
        vis_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        
        # Find the rotated bounding rectangle for the largest contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Draw the rectangle outline
        cv2.drawContours(vis_mask, [box], 0, (0, 255, 0), 2)
        
        # Calculate the center point
        center = np.mean(box, axis=0).astype(int)
        
        # Get angle of the major axis
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle = angle + 90
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Project points onto the major and minor axes
        coords = np.column_stack((x_coords, y_coords))
        coords_centered = coords - center
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Rotate points to align with axes
        rotated_coords = np.dot(coords_centered, rotation_matrix)
        
        # Find extreme points in rotated coordinates
        min_major = np.min(rotated_coords[:, 0])
        max_major = np.max(rotated_coords[:, 0])
        min_minor = np.min(rotated_coords[:, 1])
        max_minor = np.max(rotated_coords[:, 1])
        
        # Calculate length and width in pixels
        length_px = max_major - min_major
        width_px = max_minor - min_minor
        
        # Prevent division by zero
        if width_px == 0:
            width_px = 1
        
        # Convert to mm
        length_mm = length_px * self.pixel_size
        width_mm = width_px * self.pixel_size
        ratio = length_mm / width_mm
        
        # Calculate edge points for visualization
        # Length line (red)
        major_axis_pt1 = center + np.array([min_major * np.cos(angle_rad), min_major * np.sin(angle_rad)])
        major_axis_pt2 = center + np.array([max_major * np.cos(angle_rad), max_major * np.sin(angle_rad)])
        
        # Width line (blue)
        minor_axis_pt1 = center + np.array([min_minor * np.cos(angle_rad + np.pi/2), min_minor * np.sin(angle_rad + np.pi/2)])
        minor_axis_pt2 = center + np.array([max_minor * np.cos(angle_rad + np.pi/2), max_minor * np.sin(angle_rad + np.pi/2)])
        
        # Draw dimension lines
        cv2.line(vis_mask, tuple(map(int, major_axis_pt1)), tuple(map(int, major_axis_pt2)), (0, 0, 255), 2)  # Red for length
        cv2.line(vis_mask, tuple(map(int, minor_axis_pt1)), tuple(map(int, minor_axis_pt2)), (255, 0, 0), 2)  # Blue for width
        
        # Calculate exact bounding box dimensions from the rect
        # rect[1] contains (width, height) of the bounding rectangle
        # Ensure that box_length is always the larger dimension and box_width is the smaller
        dim1_px = rect[1][0]
        dim2_px = rect[1][1]
        
        if dim1_px >= dim2_px:
            box_length_px = dim1_px
            box_width_px = dim2_px
        else:
            box_length_px = dim2_px
            box_width_px = dim1_px
        
        # Convert box dimensions to mm
        box_length_mm = box_length_px * self.pixel_size
        box_width_mm = box_width_px * self.pixel_size
        
        return length_mm, width_mm, ratio, box_length_mm, box_width_mm, vis_mask

    def calculate_curvature(self, mask):
        """
        Estimate body curvature by slicing the mask along its principal axis
        and tracking the midline centroid of each slice.

        Method:
          1. Project mask pixels onto the major body axis (from minAreaRect).
          2. Divide the projection into N equal slices; compute the lateral
             centroid of each slice — these form the midline.
          3. Compute the angle at the midpoint of the midline
             (head→mid vector vs mid→tail vector).
             A straight body → angle ≈ 180°.
             A curved body  → angle < 180°.

        Returns:
          curvature_label (str) : 'No Curvature' or 'Curvature Present'
          body_angle (float)    : angle at the midpoint in degrees
                                  (180 = perfectly straight)
        """
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 'No Curvature', 180.0

        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90
        angle_rad = np.radians(angle)
        center = np.array(rect[0])

        y_coords, x_coords = np.nonzero(clean_mask)
        if len(x_coords) < 10:
            return 'No Curvature', 180.0

        coords = np.column_stack((x_coords.astype(float), y_coords.astype(float)))
        coords_centered = coords - center

        major_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        minor_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        proj_major = coords_centered @ major_dir
        proj_minor = coords_centered @ minor_dir

        N = 10
        min_proj = proj_major.min()
        max_proj = proj_major.max()
        if max_proj - min_proj < 1:
            return 'No Curvature', 180.0

        slice_width = (max_proj - min_proj) / N
        midline_points = []
        for i in range(N):
            lo = min_proj + i * slice_width
            hi = lo + slice_width
            in_slice = (proj_major >= lo) & (proj_major < hi)
            if np.sum(in_slice) < 5:
                continue
            midline_points.append([
                proj_major[in_slice].mean(),
                proj_minor[in_slice].mean()
            ])

        if len(midline_points) < 3:
            return 'No Curvature', 180.0

        midline_points = np.array(midline_points)
        head_pt = midline_points[0]
        mid_pt  = midline_points[len(midline_points) // 2]
        tail_pt = midline_points[-1]

        v1 = mid_pt - head_pt
        v2 = tail_pt - mid_pt
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 'No Curvature', 180.0

        cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(cos_a))  # 0° when perfectly straight
        body_angle = 180.0 - angle_between              # 180° when perfectly straight

        CURVATURE_THRESHOLD = 20.0  # degrees of deviation from a straight line
        if angle_between > CURVATURE_THRESHOLD:
            return 'Curvature Present', body_angle
        else:
            return 'No Curvature', body_angle

    # Also update the calculate_area method to use the cleaned mask
    def calculate_area(self, mask):
        """Calculate area using only the largest connected component"""
        import cv2
        import numpy as np
        
        # Convert mask to uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find all contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour (main region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        clean_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Calculate area from the cleaned mask
        num_pixels = np.sum(clean_mask > 0)
        area_mm2 = num_pixels * (self.pixel_size ** 2)
        return area_mm2
    
    # Update the create_overlay method to use cleaned mask
    def create_overlay(self, original_image, mask):
        """Create overlay using only the largest connected component"""
        import cv2
        import numpy as np
        
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Convert mask to uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find all contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main region)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a new mask with only the largest contour
            clean_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
            
            # Use the cleaned mask for overlay
            mask = clean_mask
        
        overlay = np.zeros_like(original_image)
        overlay[mask > 0] = [0, 0, 255]  # Red overlay for the main region only
        return cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    def classify_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.phenotype_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.classification_model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            
            # Get all classifications above threshold
            threshold = 0.2  # Adjust this threshold as needed
            classifications = []
            for idx, prob in enumerate(probs):
                confidence = prob.item()
                if confidence > threshold:
                    classifications.append((self.class_names[idx], confidence))
            
            # Sort by confidence
            classifications.sort(key=lambda x: x[1], reverse=True)
            
            return classifications
    
    def _check_yolk_sac_triple_anomaly(self, area, box_length, box_width, orientation="Ventral"):
        """
        Check yolk_sac anomalies using area, box length, and box width measurements with separate thresholds
        """
        model_data = self.control_values.get('Yolk_Sac', {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            return "No Control", 0, "No Control", "No Control", "No Control"
        
        # Get control values and thresholds for all three dimensions
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        area_threshold = orientation_data.get('area_threshold', 5.0)
        
        min_length = orientation_data.get('min_length', 0)
        max_length = orientation_data.get('max_length', 0)
        length_threshold = orientation_data.get('length_threshold', 5.0)
        
        min_width = orientation_data.get('min_width', 0)
        max_width = orientation_data.get('max_width', 0)
        width_threshold = orientation_data.get('width_threshold', 5.0)
        
        # Check each dimension with its specific threshold
        area_status, area_percentage = self._check_single_dimension(area, min_area, max_area, "area", area_threshold)
        length_status, length_percentage = self._check_single_dimension(box_length, min_length, max_length, "box_length", length_threshold)
        width_status, width_percentage = self._check_single_dimension(box_width, min_width, max_width, "box_width", width_threshold)
        
        # Determine overall status
        all_statuses = [area_status, length_status, width_status]
        
        if all(status == "No Control" for status in all_statuses):
            overall_status = "No Control"
            overall_percentage = 0
        elif any(status == "Error" for status in all_statuses):
            overall_status = "Error"
            overall_percentage = 0
        elif any(status in ["Small Anomaly", "Large Anomaly"] for status in all_statuses):
            # If any dimension is anomalous, overall is anomalous
            if any(status == "Large Anomaly" for status in all_statuses):
                overall_status = "Large Anomaly"
            else:
                overall_status = "Small Anomaly"
            # Use the minimum percentage (most problematic dimension)
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = min(valid_percentages) if valid_percentages else 0
        else:
            # All are normal
            overall_status = "Normal"
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = sum(valid_percentages) / len(valid_percentages) if valid_percentages else 100
        
        return overall_status, overall_percentage, area_status, length_status, width_status

    def _check_yolksac_ext_triple_anomaly(self, area, box_length, box_width, orientation="Ventral"):
        """
        Check yolksac_ext anomalies using area, box length, and box width measurements with separate thresholds
        """
        model_data = self.control_values.get('Yolksac_ext', {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            return "No Control", 0, "No Control", "No Control", "No Control"
        
        # Get control values and thresholds for all three dimensions
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        area_threshold = orientation_data.get('area_threshold', 5.0)
        
        min_length = orientation_data.get('min_length', 0)
        max_length = orientation_data.get('max_length', 0)
        length_threshold = orientation_data.get('length_threshold', 5.0)
        
        min_width = orientation_data.get('min_width', 0)
        max_width = orientation_data.get('max_width', 0)
        width_threshold = orientation_data.get('width_threshold', 5.0)
        
        # Check each dimension with its specific threshold
        area_status, area_percentage = self._check_single_dimension(area, min_area, max_area, "area", area_threshold)
        length_status, length_percentage = self._check_single_dimension(box_length, min_length, max_length, "box_length", length_threshold)
        width_status, width_percentage = self._check_single_dimension(box_width, min_width, max_width, "box_width", width_threshold)
        
        # Determine overall status
        all_statuses = [area_status, length_status, width_status]
        
        if all(status == "No Control" for status in all_statuses):
            overall_status = "No Control"
            overall_percentage = 0
        elif any(status == "Error" for status in all_statuses):
            overall_status = "Error"
            overall_percentage = 0
        elif any(status in ["Small Anomaly", "Large Anomaly"] for status in all_statuses):
            # If any dimension is anomalous, overall is anomalous
            if any(status == "Large Anomaly" for status in all_statuses):
                overall_status = "Large Anomaly"
            else:
                overall_status = "Small Anomaly"
            # Use the minimum percentage (most problematic dimension)
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = min(valid_percentages) if valid_percentages else 0
        else:
            # All are normal
            overall_status = "Normal"
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = sum(valid_percentages) / len(valid_percentages) if valid_percentages else 100
        
        return overall_status, overall_percentage, area_status, length_status, width_status


    def load_control_values(self):
        """Load control value ranges from JSON file"""
        try:
            with open('control_values.json', 'r') as f:
                loaded_values = json.load(f)
                
            # Check if we need to migrate from old format (class-based) to new format (model-based)
            old_format_detected = False
            
            # Check if the keys are class names instead of segmentation models
            for key in loaded_values.keys():
                if key in self.class_names and key not in self.segmentation_models.keys():
                    old_format_detected = True
                    break
            
            if old_format_detected or (loaded_values and isinstance(next(iter(loaded_values.values())), (int, float))):
                # Old format detected - migrate to new format
                print("Migrating control values from old format to new segmentation model format...")
                migrated_values = {}
                # orientations = ["Ventral", "Lateral", "Ventro Lateral"]
                orientations = [ 'Ventral' , 'latera_VentroLateral']
                
                # Create structure based on segmentation models
                for model_name in self.segmentation_models.keys():
                    migrated_values[model_name] = {}
                    for orientation in orientations:
                        if model_name == 'Pericardium':
                            migrated_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Head':
                            migrated_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolk_Sac':
                            migrated_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolksac_ext':
                            migrated_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        else:
                            # All other models now have area + box length with thresholds
                            migrated_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min': 0.0, 'max': 0.0, 'length_threshold': 5.0
                            }
                
                # Save the migrated values
                self.save_control_values(migrated_values)
                return migrated_values
            
            # New format - verify structure based on segmentation models
            # orientations = ["Ventral", "Lateral", "Ventro Lateral"]
            orientations = [ 'Ventral' , 'latera_VentroLateral']
            for model_name in self.segmentation_models.keys():
                if model_name not in loaded_values:
                    loaded_values[model_name] = {}
                for orientation in orientations:
                    if orientation not in loaded_values[model_name]:
                        if model_name == 'Pericardium':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Head':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolk_Sac':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolksac_ext':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        else:
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min': 0.0, 'max': 0.0, 'length_threshold': 5.0
                            }
                    elif not isinstance(loaded_values[model_name][orientation], dict):
                        if model_name == 'Pericardium':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Head':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolk_Sac':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        elif model_name == 'Yolksac_ext':
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                                'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                            }
                        else:
                            loaded_values[model_name][orientation] = {
                                'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                                'min': 0.0, 'max': 0.0, 'length_threshold': 5.0
                            }
                    else:
                        # Verify and add missing fields for each model type
                        if model_name == 'Pericardium':
                            # Ensure pericardium has all required fields
                            required_fields = ['min_area', 'max_area', 'area_threshold', 'min_length', 'max_length', 'length_threshold', 'min_width', 'max_width', 'width_threshold']
                            for field in required_fields:
                                if field not in loaded_values[model_name][orientation]:
                                    if 'threshold' in field:
                                        loaded_values[model_name][orientation][field] = 5.0
                                    else:
                                        loaded_values[model_name][orientation][field] = 0.0
                        elif model_name == 'Head':
                            # Ensure head has all required fields
                            required_fields = ['min_area', 'max_area', 'area_threshold', 'min_length', 'max_length', 'length_threshold', 'min_width', 'max_width', 'width_threshold']
                            for field in required_fields:
                                if field not in loaded_values[model_name][orientation]:
                                    if 'threshold' in field:
                                        loaded_values[model_name][orientation][field] = 5.0
                                    else:
                                        loaded_values[model_name][orientation][field] = 0.0
                        elif model_name == 'Yolk_Sac':
                            # Ensure yolk_sac has all required fields
                            required_fields = ['min_area', 'max_area', 'area_threshold', 'min_length', 'max_length', 'length_threshold', 'min_width', 'max_width', 'width_threshold']
                            for field in required_fields:
                                if field not in loaded_values[model_name][orientation]:
                                    if 'threshold' in field:
                                        loaded_values[model_name][orientation][field] = 5.0
                                    else:
                                        loaded_values[model_name][orientation][field] = 0.0
                        elif model_name == 'Yolksac_ext':
                            # Ensure yolksac_ext has all required fields
                            required_fields = ['min_area', 'max_area', 'area_threshold', 'min_length', 'max_length', 'length_threshold', 'min_width', 'max_width', 'width_threshold']
                            for field in required_fields:
                                if field not in loaded_values[model_name][orientation]:
                                    if 'threshold' in field:
                                        loaded_values[model_name][orientation][field] = 5.0
                                    else:
                                        loaded_values[model_name][orientation][field] = 0.0
                        else:
                            # Check if fields exist for other models (area + box length)
                            required_fields = ['min_area', 'max_area', 'area_threshold', 'min', 'max', 'length_threshold']
                            for field in required_fields:
                                if field not in loaded_values[model_name][orientation]:
                                    if 'threshold' in field:
                                        loaded_values[model_name][orientation][field] = 5.0
                                    else:
                                        loaded_values[model_name][orientation][field] = 0.0
            
            return loaded_values
            
        except FileNotFoundError:
            # Create default structure with ranges for each segmentation model and orientation
            default_values = {}
            # orientations = ["Ventral", "Lateral", "Ventro Lateral"]
            orientations = [ 'Ventral' , 'latera_VentroLateral']
            
            for model_name in self.segmentation_models.keys():
                default_values[model_name] = {}
                for orientation in orientations:
                    if model_name == 'Pericardium':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Head':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Yolk_Sac':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Yolksac_ext':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    else:
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min': 0.0, 'max': 0.0, 'length_threshold': 5.0
                        }
            
            self.save_control_values(default_values)
            return default_values
        except Exception as e:
            print(f"Error loading control values: {str(e)}")
            # Return default values if there's any other error
            default_values = {}
            # orientations = ["Ventral", "Lateral", "Ventro Lateral"]
            orientations = [ 'Ventral' , 'latera_VentroLateral']
            
            for model_name in self.segmentation_models.keys():
                default_values[model_name] = {}
                for orientation in orientations:
                    if model_name == 'Pericardium':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Head':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Yolk_Sac':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    elif model_name == 'Yolksac_ext':
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min_length': 0.0, 'max_length': 0.0, 'length_threshold': 5.0,
                            'min_width': 0.0, 'max_width': 0.0, 'width_threshold': 5.0
                        }
                    else:
                        default_values[model_name][orientation] = {
                            'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0,
                            'min': 0.0, 'max': 0.0, 'length_threshold': 5.0
                        }
            
            return default_values
    
    def _check_area_and_box_length_anomaly(self, area, box_length, model_name, orientation="Ventral"):
        """
        Check anomaly for models that use both area and box length measurements with separate thresholds
        Returns tuple with individual status for area and box length, plus overall status
        """
        model_data = self.control_values.get(model_name, {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            orientation_data = {'min_area': 0.0, 'max_area': 0.0, 'area_threshold': 5.0, 'min': 0.0, 'max': 0.0, 'length_threshold': 5.0}
        
        # Get control values and thresholds for both measurements
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        area_threshold = orientation_data.get('area_threshold', 5.0)
        
        min_length = orientation_data.get('min', 0)
        max_length = orientation_data.get('max', 0)
        length_threshold = orientation_data.get('length_threshold', 5.0)
        
        # Check each dimension with its specific threshold
        area_status, area_percentage = self._check_single_dimension(area, min_area, max_area, "area", area_threshold)
        length_status, length_percentage = self._check_single_dimension(box_length, min_length, max_length, "box_length", length_threshold)
        
        # Determine overall status
        all_statuses = [area_status, length_status]
        
        if all(status == "No Control" for status in all_statuses):
            overall_status = "No Control"
            overall_percentage = 0
        elif any(status == "Error" for status in all_statuses):
            overall_status = "Error"
            overall_percentage = 0
        elif any(status in ["Small Anomaly", "Large Anomaly"] for status in all_statuses):
            # If any dimension is anomalous, overall is anomalous
            if any(status == "Large Anomaly" for status in all_statuses):
                overall_status = "Large Anomaly"
            else:
                overall_status = "Small Anomaly"
            # Use the minimum percentage (most problematic dimension)
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = min(valid_percentages) if valid_percentages else 0
        else:
            # All are normal
            overall_status = "Normal"
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = sum(valid_percentages) / len(valid_percentages) if valid_percentages else 100
        
        return overall_status, overall_percentage, area_status, length_status

    def save_control_values(self, values):
        """Save control value ranges to JSON file"""
        with open('control_values.json', 'w') as f:
            json.dump(values, f, indent=4)

    def update_control_value(self, model_name, orientation, min_value, max_value):
        """Update control value range for a specific segmentation model and orientation"""
        if model_name not in self.control_values:
            self.control_values[model_name] = {}
        
        if orientation not in self.control_values[model_name]:
            self.control_values[model_name][orientation] = {}
        
        self.control_values[model_name][orientation] = {
            'min': float(min_value),
            'max': float(max_value)
        }
        
        self.save_control_values(self.control_values)

    
    def check_anomaly(self, box_length, box_width, model_name, orientation="Ventral", area=None, length=None, width=None):
        """
        Check if the measurements fall within the control range for the given segmentation model and orientation
        - Pericardium: Uses area, box length, AND box width comparisons
        - Head: Uses area, length, AND width comparisons
        - Yolk_Sac: Uses area, box length, AND box width comparisons
        - Yolksac_ext: Uses area, box length, AND box width comparisons
        - All others: Uses area AND box length comparisons (now with individual statuses)
        """
        try:
            # Special handling for models with triple measurements
            if model_name == 'Pericardium':
                if area is None:
                    return "Error", 0, "Error", "Error", "Error"
                return self._check_pericardium_triple_anomaly(area, box_length, box_width, orientation)
            elif model_name == 'Head':
                if area is None or length is None or width is None:
                    return "Error", 0, "Error", "Error", "Error"
                return self._check_head_triple_anomaly(area, length, width, orientation)
            elif model_name == 'Yolk_Sac':
                if area is None:
                    return "Error", 0, "Error", "Error", "Error"
                return self._check_yolk_sac_triple_anomaly(area, box_length, box_width, orientation)
            elif model_name == 'Yolksac_ext':
                if area is None:
                    return "Error", 0, "Error", "Error", "Error"
                return self._check_yolksac_ext_triple_anomaly(area, box_length, box_width, orientation)
            else:
                # All other models use area AND box length WITH INDIVIDUAL STATUSES
                if area is None:
                    return "Error", 0, "Error", "Error"
                return self._check_area_and_box_length_anomaly(area, box_length, model_name, orientation)
                
        except Exception as e:
            print(f"Error in check_anomaly: {str(e)}")
            if model_name in ['Pericardium', 'Head', 'Yolk_Sac', 'Yolksac_ext']:
                return "Error", 0, "Error", "Error", "Error"
            else:
                return "Error", 0, "Error", "Error"
            
    def _check_head_triple_anomaly(self, area, length, width, orientation="Ventral"):
        """
        Check head anomalies using area, length, and width measurements with separate thresholds
        """
        model_data = self.control_values.get('Head', {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            return "No Control", 0, "No Control", "No Control", "No Control"
        
        # Get control values and thresholds for all three dimensions
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        area_threshold = orientation_data.get('area_threshold', 5.0)
        
        min_length = orientation_data.get('min_length', 0)
        max_length = orientation_data.get('max_length', 0)
        length_threshold = orientation_data.get('length_threshold', 5.0)
        
        min_width = orientation_data.get('min_width', 0)
        max_width = orientation_data.get('max_width', 0)
        width_threshold = orientation_data.get('width_threshold', 5.0)
        
        # Check each dimension with its specific threshold
        area_status, area_percentage = self._check_single_dimension(area, min_area, max_area, "area", area_threshold)
        length_status, length_percentage = self._check_single_dimension(length, min_length, max_length, "length", length_threshold)
        width_status, width_percentage = self._check_single_dimension(width, min_width, max_width, "width", width_threshold)
        
        # Determine overall status
        all_statuses = [area_status, length_status, width_status]
        
        if all(status == "No Control" for status in all_statuses):
            overall_status = "No Control"
            overall_percentage = 0
        elif any(status == "Error" for status in all_statuses):
            overall_status = "Error"
            overall_percentage = 0
        elif any(status in ["Small Anomaly", "Large Anomaly"] for status in all_statuses):
            # If any dimension is anomalous, overall is anomalous
            if any(status == "Large Anomaly" for status in all_statuses):
                overall_status = "Large Anomaly"
            else:
                overall_status = "Small Anomaly"
            # Use the minimum percentage (most problematic dimension)
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = min(valid_percentages) if valid_percentages else 0
        else:
            # All are normal
            overall_status = "Normal"
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage]) 
                               if s not in ["No Control", "Error"]]
            overall_percentage = sum(valid_percentages) / len(valid_percentages) if valid_percentages else 100
        
        return overall_status, overall_percentage, area_status, length_status, width_status
    
    
    def _check_head_area_anomaly(self, area, orientation="Ventral"):
        """
        Check head anomalies using mask area measurement
        """
        model_data = self.control_values.get('Head', {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            return "No Control", 0
        
        # Get control values for area
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        
        # If no control values set
        if min_area == 0 and max_area == 0:
            return "No Control", 0
        
        # If only one value is set, treat as single point reference
        if min_area == 0 and max_area > 0:
            percentage = (area / max_area) * 100
            if percentage < 90:
                return "Small Anomaly", percentage
            elif percentage > 110:
                return "Large Anomaly", percentage
            return "Normal", percentage
        
        elif max_area == 0 and min_area > 0:
            percentage = (area / min_area) * 100
            if percentage < 90:
                return "Small Anomaly", percentage
            elif percentage > 110:
                return "Large Anomaly", percentage
            return "Normal", percentage
        
        # Both values set - check if within range
        if min_area <= area <= max_area:
            # Calculate position within range (0-100%)
            range_span = max_area - min_area
            if range_span > 0:
                position_in_range = ((area - min_area) / range_span) * 100
            else:
                position_in_range = 100
            return "Normal", position_in_range
        
        elif area < min_area:
            # Below minimum
            percentage_of_min = (area / min_area) * 100
            return "Small Anomaly", percentage_of_min
        
        else:  # area > max_area
            # Above maximum
            percentage_of_max = (area / max_area) * 100
            return "Large Anomaly", percentage_of_max

    def _check_pericardium_triple_anomaly(self, area, box_length, box_width, orientation="Ventral"):
        """
        Check pericardium anomalies using area, box length, and box width measurements with separate thresholds
        """
        model_data = self.control_values.get('Pericardium', {})
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            return "No Control", 0, "No Control", "No Control", "No Control"

        # Get control values and thresholds for all three dimensions
        min_area = orientation_data.get('min_area', 0)
        max_area = orientation_data.get('max_area', 0)
        area_threshold = orientation_data.get('area_threshold', 5.0)

        min_length = orientation_data.get('min_length', 0)
        max_length = orientation_data.get('max_length', 0)
        length_threshold = orientation_data.get('length_threshold', 5.0)

        min_width = orientation_data.get('min_width', 0)
        max_width = orientation_data.get('max_width', 0)
        width_threshold = orientation_data.get('width_threshold', 5.0)

        # If all minimum thresholds are 0 and all measurements are 0, the mask was not found
        # and the user has indicated no minimum size requirement - treat as no anomaly
        if min_area == 0 and min_length == 0 and min_width == 0 and area == 0 and box_length == 0 and box_width == 0:
            return "Normal", 100, "Normal", "Normal", "Normal"

        # Check each dimension with its specific threshold
        area_status, area_percentage = self._check_single_dimension(area, min_area, max_area, "area", area_threshold)
        length_status, length_percentage = self._check_single_dimension(box_length, min_length, max_length, "box_length", length_threshold)
        width_status, width_percentage = self._check_single_dimension(box_width, min_width, max_width, "box_width", width_threshold)

        # If minimum is set to 0, there is no lower bound - do not flag Small Anomaly for pericardium
        if min_area == 0 and area_status == "Small Anomaly":
            area_status = "Normal"
            area_percentage = (area / max_area) * 100 if max_area > 0 else 100
        if min_length == 0 and length_status == "Small Anomaly":
            length_status = "Normal"
            length_percentage = (box_length / max_length) * 100 if max_length > 0 else 100
        if min_width == 0 and width_status == "Small Anomaly":
            width_status = "Normal"
            width_percentage = (box_width / max_width) * 100 if max_width > 0 else 100

        # Determine overall status
        all_statuses = [area_status, length_status, width_status]

        if all(status == "No Control" for status in all_statuses):
            overall_status = "No Control"
            overall_percentage = 0
        elif any(status == "Error" for status in all_statuses):
            overall_status = "Error"
            overall_percentage = 0
        elif any(status in ["Small Anomaly", "Large Anomaly"] for status in all_statuses):
            # If any dimension is anomalous, overall is anomalous
            if any(status == "Large Anomaly" for status in all_statuses):
                overall_status = "Large Anomaly"
            else:
                overall_status = "Small Anomaly"
            # Use the minimum percentage (most problematic dimension)
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage])
                               if s not in ["No Control", "Error"]]
            overall_percentage = min(valid_percentages) if valid_percentages else 0
        else:
            # All are normal
            overall_status = "Normal"
            valid_percentages = [p for s, p in zip(all_statuses, [area_percentage, length_percentage, width_percentage])
                               if s not in ["No Control", "Error"]]
            overall_percentage = sum(valid_percentages) / len(valid_percentages) if valid_percentages else 100

        return overall_status, overall_percentage, area_status, length_status, width_status

    def _check_single_dimension(self, measurement, min_val, max_val, dimension_name, threshold=5.0):
        """
        Check a single dimension against its control values with custom threshold
        """
        # If no control values set
        if min_val == 0 and max_val == 0:
            return "No Control", 0
        
        # Convert threshold to decimal (e.g., 5.0 becomes 0.05)
        threshold_decimal = threshold / 100.0
        
        # If only one value is set, treat as single point reference with custom threshold
        if (min_val == 0 and max_val > 0) or (max_val == 0 and min_val > 0):
            reference_val = max_val if max_val > 0 else min_val
            lower_threshold = reference_val * (1 - threshold_decimal)
            upper_threshold = reference_val * (1 + threshold_decimal)
            
            if lower_threshold <= measurement <= upper_threshold:
                percentage = (measurement / reference_val) * 100
                return "Normal", percentage
            elif measurement < lower_threshold:
                percentage = (measurement / reference_val) * 100
                return "Small Anomaly", percentage
            else:  # measurement > upper_threshold
                percentage = (measurement / reference_val) * 100
                return "Large Anomaly", percentage
        
        # Both values set - create expanded range with custom threshold
        else:
            expanded_min = min_val * (1 - threshold_decimal)
            expanded_max = max_val * (1 + threshold_decimal)
            
            if expanded_min <= measurement <= expanded_max:
                # Within expanded range (including custom tolerance) - Normal
                if min_val <= measurement <= max_val:
                    range_span = max_val - min_val
                    if range_span > 0:
                        position_in_range = ((measurement - min_val) / range_span) * 100
                    else:
                        position_in_range = 100
                else:
                    if measurement < min_val:
                        position_in_range = (measurement / min_val) * 100
                    else:
                        position_in_range = (measurement / max_val) * 100
                return "Normal", position_in_range
            elif measurement < expanded_min:
                percentage_of_min = (measurement / min_val) * 100
                return "Small Anomaly", percentage_of_min
            else:  # measurement > expanded_max
                percentage_of_max = (measurement / max_val) * 100
                return "Large Anomaly", percentage_of_max
    
    def _check_single_measurement_anomaly(self, size, model_name, orientation):
        """
        Check anomaly for models that use single measurement (box length)
        """
        # Safely get control values with fallback
        model_data = self.control_values.get(model_name, {})
        
        # Handle case where model_data might still be in old format
        if isinstance(model_data, (int, float)):
            # Old format fallback
            control_value = float(model_data)
            if control_value == 0:
                return "No Control", 0
            percentage = (size / control_value) * 100
            if percentage < 95:
                return "Small Anomaly", percentage
            elif percentage > 105:
                return "Large Anomaly", percentage
            return "Normal", percentage
        
        # New format
        orientation_data = model_data.get(orientation, {})
        
        # Handle case where orientation_data might be missing or invalid
        if not isinstance(orientation_data, dict):
            orientation_data = {'min': 0.0, 'max': 0.0}
        
        min_val = orientation_data.get('min', 0)
        max_val = orientation_data.get('max', 0)
        
        # If no control values set
        if min_val == 0 and max_val == 0:
            return "No Control", 0
        
        if (min_val == 0 and max_val > 0) or (max_val == 0 and min_val > 0):
            reference_val = max_val if max_val > 0 else min_val
            percentage = (size / reference_val) * 100
            if percentage < 95:
                return "Small Anomaly", percentage
            elif percentage > 105:
                return "Large Anomaly", percentage
            return "Normal", percentage
        
        # Both values set - check if within range
        if min_val <= size <= max_val:
            # Calculate position within range (0-100%)
            range_span = max_val - min_val
            if range_span > 0:
                position_in_range = ((size - min_val) / range_span) * 100
            else:
                position_in_range = 100
            return "Normal", position_in_range
        
        elif size < min_val:
            # Below minimum
            percentage_of_min = (size / min_val) * 100
            return "Small Anomaly", percentage_of_min
        
        else:  # size > max_val
            # Above maximum
            percentage_of_max = (size / max_val) * 100
            return "Large Anomaly", percentage_of_max
    
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            print("psutil not installed - memory monitoring disabled")
            return 0
    
    def force_cleanup(self):
        """Force garbage collection and GPU cleanup"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory_threshold(self, threshold_mb=4000):
        """Check if memory usage exceeds threshold (default 4GB)"""
        current_memory = self.get_memory_usage_mb()
        if current_memory > threshold_mb:
            print(f"Memory usage high: {current_memory:.1f} MB")
            return True
        return False
        
class DanioAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = ImageAnalyzer()
        self.current_index = -1
        self.results = []
        self.processed_images = []
        self.current_folder = ""  # Add this line
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("DanioV16 Analyzer")
        self.setGeometry(100, 100, 1500, 800)
    
        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
    
        # Segmentation model buttons
        segmentation_panel = QWidget()
        segmentation_layout = QHBoxLayout(segmentation_panel)
        segmentation_layout.addWidget(QLabel("Segmentation Models:"))
        
        self.segmentation_buttons = {}
        for model_name in self.analyzer.segmentation_models.keys():
            btn = QPushButton(model_name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, name=model_name: self.change_segmentation_model(name))
            self.segmentation_buttons[model_name] = btn
            segmentation_layout.addWidget(btn)
        
        if self.segmentation_buttons:
            self.segmentation_buttons['Whole Larva'].setChecked(True)
            
        layout.addWidget(segmentation_panel)
    
        # Measurement display
        self.measurement_label = QLabel()
        self.measurement_label.setAlignment(Qt.AlignCenter)
        self.measurement_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.measurement_label)
    
        # Image display area
        display_layout = QHBoxLayout()
        
        # Original image
        self.original_label = QLabel()
        self.original_label.setMinimumSize(500, 400)
        display_layout.addWidget(self.original_label)
        
        # Mask
        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(500, 400)
        display_layout.addWidget(self.mask_label)
        
        # Overlay
        self.overlay_label = QLabel()
        self.overlay_label.setMinimumSize(500, 400)
        display_layout.addWidget(self.overlay_label)
        
        layout.addLayout(display_layout)
    
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
    
        # Control buttons
        control_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Process Folder")
        self.folder_btn.clicked.connect(self.process_folder)
        control_layout.addWidget(self.folder_btn)

        self.control_btn = QPushButton("Set Control Values")
        self.control_btn.clicked.connect(self.show_control_dialog)
        control_layout.addWidget(self.control_btn)

        layout.addLayout(control_layout)

        from PyQt5.QtWidgets import QCheckBox
        self.include_phenotype_checkbox = QCheckBox("Include phenotype in saved images")
        layout.addWidget(self.include_phenotype_checkbox)
        self.include_orientation_checkbox = QCheckBox("Include orientation in saved images")
        layout.addWidget(self.include_orientation_checkbox)

    def show_control_dialog(self):
        dialog = ControlValueDialog(self.analyzer, self)
        if dialog.exec_() == QDialog.Accepted:
            # Update display if needed
            if self.current_index >= 0:
                self.display_current_image()
                
  
    def _generate_display_data_for_image(self, image_path, model_names, vis_cache=None):
        """
        Return display data for an image not held in the in-memory display cache.

        Fast path  : decode compressed PNG bytes from vis_cache (built during processing).
        Slow path  : re-run ML segmentation (fallback if cache entry is missing).
        """
        display_data = {'original': None, 'dimension_masks': {}, 'overlays': {}}
        filename = os.path.basename(image_path)

        # ── Fast path: decode from in-memory cache ───────────────────────────────
        if vis_cache and filename in vis_cache:
            entry = vis_cache[filename]
            if entry.get('original') is not None:
                orig_buf = np.frombuffer(entry['original'], np.uint8)
                display_data['original'] = cv2.imdecode(orig_buf, cv2.IMREAD_COLOR)
            for model_name in model_names:
                if model_name in entry.get('masks', {}):
                    vis_buf = np.frombuffer(entry['masks'][model_name], np.uint8)
                    display_data['dimension_masks'][model_name] = cv2.imdecode(vis_buf, cv2.IMREAD_COLOR)
                if model_name in entry.get('overlays', {}):
                    ovl_buf = np.frombuffer(entry['overlays'][model_name], np.uint8)
                    display_data['overlays'][model_name] = cv2.imdecode(ovl_buf, cv2.IMREAD_COLOR)
            if display_data['original'] is not None:
                return display_data

        # ── Slow path: re-run segmentation ──────────────────────────────────────
        print(f"Cache miss for {filename} — re-running segmentation (slow fallback).")
        for model_name in model_names:
            try:
                self.analyzer.current_segmentation = model_name
                mask, original_image, _, _, _, _, _, vis_mask = \
                    self.analyzer.segment_image(image_path)
                if display_data['original'] is None:
                    display_data['original'] = original_image
                display_data['dimension_masks'][model_name] = vis_mask
                display_data['overlays'][model_name] = \
                    self.analyzer.create_overlay(original_image, mask)
                del mask
                self.analyzer.clear_gpu_cache()
            except Exception as e:
                print(f"Re-segmentation error ({model_name}, {filename}): {str(e)}")

        return display_data if display_data['original'] is not None else None

    def _create_composite_image(self, result, display_data, models_to_show,
                                include_phenotype=False, include_orientation=False):
        """
        Build a composite BGR numpy image for saving to anomaly / normal folders.

        Layout per model (stacked vertically):
            [optional header strip — phenotype and/or orientation]
            [original image | mask+bbox | overlay]   ← PANEL_H px tall
            [footer: model name, Area, Length, Width, anomaly status with dimension detail]

        Parameters
        ----------
        result             : result dict for this image
        display_data       : {'original': ndarray, 'dimension_masks': {model: ndarray},
                               'overlays': {model: ndarray}}
        models_to_show     : list of model names to include (one row each)
        include_phenotype  : if True, show phenotype in the header strip
        include_orientation: if True, show orientation in the header strip
        """
        PANEL_H  = 400
        FOOTER_H = 72
        HEADER_H = 38
        BG_DARK      = (30,  30,  30)
        COLOR_WHITE  = (220, 220, 220)
        COLOR_GREEN  = (60,  200, 60)
        COLOR_RED    = (60,  60,  220)
        FONT         = cv2.FONT_HERSHEY_SIMPLEX

        def resize_to_height(img, h):
            ih, iw = img.shape[:2]
            nw = max(1, int(iw * h / ih))
            return cv2.resize(img, (nw, h))

        rows = []   # list of numpy arrays; panels and footers alternate

        for model_name in models_to_show:
            original = display_data.get('original')
            vis_mask = display_data.get('dimension_masks', {}).get(model_name)
            overlay  = display_data.get('overlays', {}).get(model_name)

            if original is None or vis_mask is None or overlay is None:
                continue

            # Convert PIL Image to BGR numpy (same logic as display_image in the GUI)
            if isinstance(original, Image.Image):
                original = np.array(original)
                if len(original.shape) == 2:
                    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

            orig_r = resize_to_height(original, PANEL_H)
            mask_r = resize_to_height(vis_mask,  PANEL_H)
            over_r = resize_to_height(overlay,   PANEL_H)

            panel_row = np.hstack([orig_r, mask_r, over_r])
            row_w = panel_row.shape[1]
            rows.append(panel_row)

            # ── footer ──────────────────────────────────────────────
            area       = result.get(f'{model_name}_Area',       -1)
            box_length = result.get(f'{model_name}_Box_Length', -1)
            box_width  = result.get(f'{model_name}_Box_Width',  -1)
            status     = result.get(f'{model_name}_Status',     'Normal')
            percentage = result.get(f'{model_name}_Percentage',  0)

            area_str = f"{area:.2f} mm2"       if area       >= 0 else "N/A"
            len_str  = f"{box_length:.2f} mm"  if box_length >= 0 else "N/A"
            wid_str  = f"{box_width:.2f} mm"   if box_width  >= 0 else "N/A"

            footer = np.full((FOOTER_H, row_w, 3), BG_DARK, dtype=np.uint8)

            line1 = f"{model_name}  |  Area: {area_str}  |  Length: {len_str}  |  Width: {wid_str}"
            if model_name == 'Whole Larva':
                curvature_str = result.get('Whole Larva_Curvature', '')
                if curvature_str and curvature_str not in ('N/A', 'Error', ''):
                    line1 += f"  |  {curvature_str}"
            cv2.putText(footer, line1, (10, 26), FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

            if status in ('Small Anomaly', 'Large Anomaly'):
                # Look up control values to compute per-dimension deviation percentages
                orientation_used = result.get(f'{model_name}_Used_Orientation', 'Ventral')
                ctrl = self.analyzer.control_values.get(model_name, {}).get(orientation_used, {})

                def _dim_pct(measured, min_v, max_v, dim_status):
                    """Return % deviation from the relevant control boundary."""
                    if dim_status == 'Small Anomaly':
                        ref = min_v if min_v > 0 else max_v
                        return (1 - measured / ref) * 100 if ref > 0 else None
                    elif dim_status == 'Large Anomaly':
                        ref = max_v if max_v > 0 else min_v
                        return (measured / ref - 1) * 100 if ref > 0 else None
                    return None

                dim_parts = []

                # Area
                area_st = result.get(f'{model_name}_Area_Status', '')
                if area_st in ('Small Anomaly', 'Large Anomaly'):
                    pct = _dim_pct(result.get(f'{model_name}_Area', 0),
                                   ctrl.get('min_area', 0), ctrl.get('max_area', 0), area_st)
                    direction = 'SMALLER' if area_st == 'Small Anomaly' else 'LARGER'
                    if pct is not None:
                        dim_parts.append(f"Area: {pct:.1f}% {direction}")

                # Length — key differs by model type
                len_st = result.get(f'{model_name}_Length_Status',
                         result.get(f'{model_name}_Box_Length_Status', ''))
                if len_st in ('Small Anomaly', 'Large Anomaly'):
                    len_val = result.get(f'{model_name}_Length',
                              result.get(f'{model_name}_Box_Length', 0))
                    min_len = ctrl.get('min_length', ctrl.get('min', 0))
                    max_len = ctrl.get('max_length', ctrl.get('max', 0))
                    pct = _dim_pct(len_val, min_len, max_len, len_st)
                    direction = 'SMALLER' if len_st == 'Small Anomaly' else 'LARGER'
                    if pct is not None:
                        dim_parts.append(f"Length: {pct:.1f}% {direction}")

                # Width
                wid_st = result.get(f'{model_name}_Width_Status', '')
                if wid_st in ('Small Anomaly', 'Large Anomaly'):
                    wid_val = result.get(f'{model_name}_Box_Width',
                              result.get(f'{model_name}_Width', 0))
                    pct = _dim_pct(wid_val, ctrl.get('min_width', 0), ctrl.get('max_width', 0), wid_st)
                    direction = 'SMALLER' if wid_st == 'Small Anomaly' else 'LARGER'
                    if pct is not None:
                        dim_parts.append(f"Width: {pct:.1f}% {direction}")

                if dim_parts:
                    line2 = "  |  ".join(dim_parts) + " than control larva"
                else:
                    direction = "SMALLER" if status == 'Small Anomaly' else "LARGER"
                    line2 = f"{abs(percentage):.1f}% {direction} than control larva"
                color2 = COLOR_RED
            else:
                line2  = "NORMAL  -  within control range"
                color2 = COLOR_GREEN

            cv2.putText(footer, line2, (10, 58), FONT, 0.58, color2, 1, cv2.LINE_AA)
            rows.append(footer)

        if not rows:
            return None

        # Normalise all rows to the same width (first panel row is the reference)
        ref_w = rows[0].shape[1]
        normalised = []
        for r in rows:
            if r.shape[1] != ref_w:
                r = cv2.resize(r, (ref_w, r.shape[0]))
            normalised.append(r)

        # Optional header strip at the very top
        if include_phenotype or include_orientation:
            header = np.full((HEADER_H, ref_w, 3), BG_DARK, dtype=np.uint8)
            parts = []
            if include_phenotype:
                phenotype  = result.get('Primary_Phenotype',           'N/A')
                pheno_conf = result.get('Primary_Phenotype_Confidence',  0)
                parts.append(f"Phenotype: {phenotype} ({pheno_conf:.2f})")
            if include_orientation:
                orientation = result.get('Primary_Orientation',           'N/A')
                orient_conf = result.get('Primary_Orientation_Confidence', 0)
                parts.append(f"Orientation: {orientation} ({orient_conf:.2f})")
            htext = "   |   ".join(parts)
            cv2.putText(header, htext, (10, 26), FONT, 0.56, COLOR_WHITE, 1, cv2.LINE_AA)
            normalised.insert(0, header)

        return np.vstack(normalised)

    # Enhanced create_anomaly_folders method for DanioAnalyzerGUI class
    # Complete fixed create_anomaly_folders method for DanioAnalyzerGUI class
    def create_anomaly_folders(self, folder_path, results, processed_images=None,
                               include_phenotype=False, include_orientation=False,
                               vis_cache=None):
        """
        Create folders for images with anomalies and copy the images to respective folders
        Creates both individual anomaly folders and combination folders for multiple anomalies
        Also creates a folder for normal fish with no anomalies

        Args:
            folder_path:         The original folder path where images are located
            results:             List of result dictionaries from processing
            processed_images:    List of display data dicts (original, dimension_masks, overlays)
            include_phenotype:   Whether to show phenotype in the header strip
            include_orientation: Whether to show orientation in the header strip
            vis_cache:           In-memory dict {filename: {'original': bytes, 'masks': {}, 'overlays': {}}}
        """
        import shutil

        try:
            # Build lookups for composite image generation
            result_lookup = {r['Image']: r for r in results}
            display_lookup = {}
            if processed_images:
                for i, disp in enumerate(processed_images):
                    if i < len(results):
                        display_lookup[results[i]['Image']] = disp

            # Cache for images not in display_lookup (re-segmented on demand, once per image)
            generated_display_cache = {}
            all_model_names = list(self.analyzer.segmentation_models.keys())
            # Track individual anomaly folders
            individual_anomaly_folders = {}
            individual_anomaly_counts = {}
            
            # Track combination anomaly folders
            combination_anomaly_folders = {}
            combination_anomaly_counts = {}
            
            # Track normal fish folder
            normal_folder = None
            normal_count = 0

            # # Track pericardium skipped folder
            # pericardium_skipped_folder = None
            # pericardium_skipped_count = 0
            # pericardium_skipped_images = []

            # Initialize lists for categorizing images
            image_anomalies = {}  # {image_name: [list_of_anomalous_models]}
            normal_images = []    # List of images with no anomalies - THIS LINE IS CRUCIAL

            # First pass: collect all anomalies per image and pericardium skipped images
            for result in results:
                image_name = result['Image']
                image_path = os.path.join(folder_path, image_name)
                
                # Check if the image file exists
                if not os.path.exists(image_path):
                    continue
                
                anomalous_models = []

                # Check for pericardium skipped status
                pericardium_status = result.get('Pericardium_Status', 'Normal')
                # if pericardium_status == 'Skipped - Ventral':
                #     pericardium_skipped_images.append(image_name)

                # Check each segmentation model for anomalies
                for model_name in self.analyzer.segmentation_models.keys():
                    status_key = f'{model_name}_Status'
                    status = result.get(status_key, 'Normal')

                    if status in ['Small Anomaly', 'Large Anomaly']:
                        anomalous_models.append(model_name)

                # Store anomalous models for this image or mark as normal
                if anomalous_models:
                    image_anomalies[image_name] = anomalous_models
                else:
                    # Image has no anomalies
                    normal_images.append(image_name)
            
            # Second pass: create folders and copy images for anomalies
            for image_name, anomalous_models in image_anomalies.items():
                image_path = os.path.join(folder_path, image_name)
                
                # Create individual anomaly folders and copy images
                for model_name in anomalous_models:
                    # Create individual folder if it doesn't exist
                    if model_name not in individual_anomaly_folders:
                        individual_folder_path = os.path.join(folder_path, f"{model_name}_Anomalies")
                        os.makedirs(individual_folder_path, exist_ok=True)
                        individual_anomaly_folders[model_name] = individual_folder_path
                        individual_anomaly_counts[model_name] = 0
                    
                    # Save composite (or fallback copy) to individual anomaly folder
                    individual_destination = os.path.join(individual_anomaly_folders[model_name], image_name)
                    try:
                        saved = False
                        r = result_lookup.get(image_name)
                        disp = display_lookup.get(image_name)
                        if disp is None or disp.get('original') is None:
                            if image_name not in generated_display_cache:
                                generated_display_cache[image_name] = \
                                    self._generate_display_data_for_image(image_path, all_model_names, vis_cache)
                            disp = generated_display_cache.get(image_name)
                        if disp is not None and r is not None and disp.get('original') is not None:
                            try:
                                composite = self._create_composite_image(r, disp, [model_name], include_phenotype, include_orientation)
                                if composite is not None:
                                    saved = bool(cv2.imwrite(individual_destination, composite))
                            except Exception as comp_e:
                                print(f"Composite error for {image_name}/{model_name}: {str(comp_e)}")
                        if not saved:
                            shutil.copy2(image_path, individual_destination)
                        individual_anomaly_counts[model_name] += 1
                    except Exception as e:
                        print(f"Error saving {image_name} to {model_name} individual folder: {str(e)}")
                
                # Create combination folder if multiple anomalies exist
                if len(anomalous_models) > 1:
                    # Sort model names for consistent folder naming
                    sorted_models = sorted(anomalous_models)
                    combination_key = "_".join(sorted_models)
                    
                    # Create combination folder if it doesn't exist
                    if combination_key not in combination_anomaly_folders:
                        combination_folder_name = f"{combination_key}_Anomalies"
                        combination_folder_path = os.path.join(folder_path, combination_folder_name)
                        os.makedirs(combination_folder_path, exist_ok=True)
                        combination_anomaly_folders[combination_key] = combination_folder_path
                        combination_anomaly_counts[combination_key] = 0
                    
                    # Save composite (or fallback copy) to combination anomaly folder
                    combination_destination = os.path.join(combination_anomaly_folders[combination_key], image_name)
                    try:
                        saved = False
                        r = result_lookup.get(image_name)
                        disp = display_lookup.get(image_name)
                        if disp is None or disp.get('original') is None:
                            if image_name not in generated_display_cache:
                                generated_display_cache[image_name] = \
                                    self._generate_display_data_for_image(image_path, all_model_names, vis_cache)
                            disp = generated_display_cache.get(image_name)
                        if disp is not None and r is not None and disp.get('original') is not None:
                            try:
                                composite = self._create_composite_image(r, disp, sorted_models, include_phenotype, include_orientation)
                                if composite is not None:
                                    saved = bool(cv2.imwrite(combination_destination, composite))
                            except Exception as comp_e:
                                print(f"Composite error for {image_name} combination: {str(comp_e)}")
                        if not saved:
                            shutil.copy2(image_path, combination_destination)
                        combination_anomaly_counts[combination_key] += 1
                    except Exception as e:
                        print(f"Error saving {image_name} to combination folder {combination_key}: {str(e)}")
            
            # Third pass: create normal fish folder and copy normal images
            if normal_images:
                normal_folder_path = os.path.join(folder_path, "No_Anomaly")
                os.makedirs(normal_folder_path, exist_ok=True)
                normal_folder = normal_folder_path

                all_models = list(self.analyzer.segmentation_models.keys())
                for image_name in normal_images:
                    image_path = os.path.join(folder_path, image_name)
                    normal_destination = os.path.join(normal_folder_path, image_name)
                    try:
                        saved = False
                        r = result_lookup.get(image_name)
                        disp = display_lookup.get(image_name)
                        if disp is None or disp.get('original') is None:
                            if image_name not in generated_display_cache:
                                generated_display_cache[image_name] = \
                                    self._generate_display_data_for_image(image_path, all_model_names, vis_cache)
                            disp = generated_display_cache.get(image_name)
                        if disp is not None and r is not None and disp.get('original') is not None:
                            try:
                                composite = self._create_composite_image(r, disp, all_models, include_phenotype, include_orientation)
                                if composite is not None:
                                    saved = bool(cv2.imwrite(normal_destination, composite))
                            except Exception as comp_e:
                                print(f"Composite error for {image_name} (No_Anomaly): {str(comp_e)}")
                        if not saved:
                            shutil.copy2(image_path, normal_destination)
                        normal_count += 1
                    except Exception as e:
                        print(f"Error saving {image_name} to No_Anomaly folder: {str(e)}")

            # Fourth pass: create pericardium skipped folder and copy skipped images
            # if pericardium_skipped_images:
            #     pericardium_skipped_folder_path = os.path.join(folder_path, "Pericardium_Skipped")
            #     os.makedirs(pericardium_skipped_folder_path, exist_ok=True)
            #     pericardium_skipped_folder = pericardium_skipped_folder_path

            #     for image_name in pericardium_skipped_images:
            #         image_path = os.path.join(folder_path, image_name)
            #         skipped_destination = os.path.join(pericardium_skipped_folder_path, image_name)
            #         try:
            #             shutil.copy2(image_path, skipped_destination)
            #             pericardium_skipped_count += 1
            #         except Exception as e:
            #             print(f"Error copying {image_name} to Pericardium_Skipped folder: {str(e)}")

            # Print summary of folders created
            print("\n=== ANOMALY FOLDERS SUMMARY ===")

            if normal_folder:
                print(f"\nNormal fish folder created:")
                print(f"  No_Anomaly: {normal_count} images")

            # if pericardium_skipped_folder:
            #     print(f"\nPericardium skipped folder created:")
            #     print(f"  Pericardium_Skipped: {pericardium_skipped_count} images (Ventral orientation)")

            if individual_anomaly_folders:
                print("\nIndividual anomaly folders created:")
                for model_name, count in individual_anomaly_counts.items():
                    print(f"  {model_name}_Anomalies: {count} images")
            
            if combination_anomaly_folders:
                print("\nCombination anomaly folders created:")
                for combination_key, count in combination_anomaly_counts.items():
                    models_list = combination_key.replace("_", " + ")
                    print(f"  {combination_key}_Anomalies: {count} images ({models_list})")

            # if not individual_anomaly_folders and not combination_anomaly_folders and not normal_folder and not pericardium_skipped_folder:
            #     print("No folders created - no data processed successfully")
            
            # Create summary statistics
            total_images_processed = len(results)
            total_images_with_anomalies = len(image_anomalies)
            total_normal_images = len(normal_images)
            single_anomaly_images = sum(1 for models in image_anomalies.values() if len(models) == 1)
            multiple_anomaly_images = sum(1 for models in image_anomalies.values() if len(models) > 1)
            
            print(f"\n=== CLASSIFICATION STATISTICS ===")
            print(f"Total images processed: {total_images_processed}")
            print(f"Normal images (no anomalies): {total_normal_images} ({(total_normal_images/total_images_processed*100):.1f}%)")
            print(f"Images with anomalies: {total_images_with_anomalies} ({(total_images_with_anomalies/total_images_processed*100):.1f}%)")
            print(f"  - Single anomaly: {single_anomaly_images}")
            print(f"  - Multiple anomalies: {multiple_anomaly_images}")
            # print(f"Pericardium skipped (Ventral orientation): {len(pericardium_skipped_images)} ({(len(pericardium_skipped_images)/total_images_processed*100):.1f}%)")
            
            if multiple_anomaly_images > 0:
                print("\nMultiple anomaly breakdown:")
                anomaly_combination_stats = {}
                for models in image_anomalies.values():
                    if len(models) > 1:
                        sorted_combination = "_".join(sorted(models))
                        anomaly_combination_stats[sorted_combination] = anomaly_combination_stats.get(sorted_combination, 0) + 1
                
                for combination, count in sorted(anomaly_combination_stats.items()):
                    models_display = combination.replace("_", " + ")
                    print(f"  {models_display}: {count} images")
            
            # Create a detailed report file - CORRECTED CALL WITH ALL PARAMETERS
            # self.create_anomaly_report(folder_path, image_anomalies, normal_images, individual_anomaly_counts, combination_anomaly_counts, pericardium_skipped_images)
            self.create_anomaly_report(folder_path, image_anomalies, normal_images, individual_anomaly_counts, combination_anomaly_counts)

            # Release in-memory cache now that all composites have been saved
            vis_cache = None

        except Exception as e:
            print(f"Error creating anomaly folders: {str(e)}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
    
    # Also add the create_anomaly_report method with the correct signature
    def create_anomaly_report(self, folder_path, image_anomalies, normal_images, individual_counts, combination_counts, pericardium_skipped_images=None):
        """
        Create a detailed text report of all anomalies found

        Args:
            folder_path: The folder path where to save the report
            image_anomalies: Dictionary of {image_name: [anomalous_models]}
            normal_images: List of normal images with no anomalies
            individual_counts: Dictionary of individual anomaly counts
            combination_counts: Dictionary of combination anomaly counts
            # pericardium_skipped_images: List of images where pericardium was skipped (Ventral orientation)
        """
        if pericardium_skipped_images is None:
            pericardium_skipped_images = []
        try:
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(folder_path, f"complete_analysis_report_{timestamp}.txt")
            
            total_images = len(self.results)
            anomaly_images = len(image_anomalies)
            normal_count = len(normal_images)
            # pericardium_skipped_count = len(pericardium_skipped_images)

            with open(report_file, 'w') as f:
                f.write("ZEBRAFISH COMPLETE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Summary statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total images analyzed: {total_images}\n")
                f.write(f"Normal images (no anomalies): {normal_count} ({(normal_count/total_images*100):.1f}%)\n")
                f.write(f"Images with anomalies: {anomaly_images} ({(anomaly_images/total_images*100):.1f}%)\n")
                # f.write(f"Pericardium skipped (Ventral orientation): {pericardium_skipped_count} ({(pericardium_skipped_count/total_images*100):.1f}%)\n")
                f.write(f"Overall anomaly rate: {(anomaly_images/total_images*100):.1f}%\n\n")
                
                # Individual anomaly counts
                if individual_counts:
                    f.write("INDIVIDUAL ANOMALY COUNTS\n")
                    f.write("-" * 30 + "\n")
                    for model_name, count in sorted(individual_counts.items()):
                        rate = (count/total_images*100)
                        f.write(f"{model_name}: {count} images ({rate:.1f}%)\n")
                    f.write("\n")
                
                # Combination anomaly counts
                if combination_counts:
                    f.write("COMBINATION ANOMALY COUNTS\n")
                    f.write("-" * 30 + "\n")
                    for combination, count in sorted(combination_counts.items()):
                        models_display = combination.replace("_", " + ")
                        rate = (count/total_images*100)
                        f.write(f"{models_display}: {count} images ({rate:.1f}%)\n")
                    f.write("\n")

                # Pericardium skipped images (Ventral orientation)
                # if pericardium_skipped_images:
                #     f.write("PERICARDIUM SKIPPED IMAGES (VENTRAL ORIENTATION)\n")
                #     f.write("-" * 50 + "\n")
                #     f.write(f"Total: {pericardium_skipped_count} images\n")
                #     f.write("Images:\n")
                #     for img_name in sorted(pericardium_skipped_images):
                #         f.write(f"  - {img_name}\n")
                #     f.write("\n")

            print(f"\nComplete analysis report saved to: {os.path.basename(report_file)}")
            
        except Exception as e:
            print(f"Error creating analysis report: {str(e)}")
        
        
        # Enhanced method to also create an Excel summary of anomalies
        def create_anomaly_excel_summary(self, folder_path, image_anomalies):
            """
            Create an Excel summary of all anomalies for easy analysis
            """
            try:
                import pandas as pd
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file = os.path.join(folder_path, f"anomaly_summary_{timestamp}.xlsx")
                
                # Prepare data for Excel
                summary_data = []
                
                for image_name, anomalous_models in image_anomalies.items():
                    row = {'Image': image_name, 'Anomaly_Count': len(anomalous_models)}
                    
                    # Add individual model columns
                    for model_name in self.analyzer.segmentation_models.keys():
                        row[f'{model_name}_Anomaly'] = 'Yes' if model_name in anomalous_models else 'No'
                    
                    # Add combination description
                    if len(anomalous_models) > 1:
                        row['Combination_Type'] = "_".join(sorted(anomalous_models))
                    else:
                        row['Combination_Type'] = 'Single'
                    
                    summary_data.append(row)
                
                # Create DataFrame and save
                df = pd.DataFrame(summary_data)
                df.to_excel(excel_file, index=False)
                
                print(f"Excel anomaly summary saved to: {os.path.basename(excel_file)}")
                
            except Exception as e:
                print(f"Error creating Excel anomaly summary: {str(e)}")

# Add this call to the end of the enhanced create_anomaly_folders method (before the except block):
# self.create_anomaly_excel_summary(folder_path, image_anomalies)

    
    # Enhanced process_folder for 1000+ images
    
    
    def process_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder_path:
            return
     
        try:
            image_files = sorted(
                [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))],
                key=_natural_sort_key)

            if not image_files:
                QMessageBox.warning(self, "Warning", "No image files found.")
                return
            
            total_images = len(image_files)
            is_large_batch = total_images >= LARGE_BATCH_THRESHOLD
            
            # Adjust settings for large batches
            if is_large_batch:
                batch_size = BATCH_SIZE_LARGE
                max_display = MAX_DISPLAY_IMAGES_LARGE
                
                # Ask user about processing approach for large batches
                reply = QMessageBox.question(
                    self, "Large Batch Detected", 
                    f"Processing {total_images} images.\n\n"
                    f"For optimal performance:\n"
                    f"• Batch size: {batch_size} images\n"
                    f"• GUI display: First {max_display} images only\n"
                    f"• All results saved to Excel\n\n"
                    f"Continue with optimized settings?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.No:
                    # Allow user to customize
                    from PyQt5.QtWidgets import QInputDialog
                    batch_size, ok1 = QInputDialog.getInt(
                        self, "Batch Size", "Batch size:", batch_size, 10, 100, 5
                    )
                    if ok1:
                        max_display, ok2 = QInputDialog.getInt(
                            self, "Display Limit", "Max images for GUI:", max_display, 10, 200, 10
                        )
                        if not ok2:
                            return
                    else:
                        return
                elif reply == QMessageBox.Cancel:
                    return
            else:
                batch_size = BATCH_SIZE_NORMAL
                max_display = MAX_DISPLAY_IMAGES_NORMAL
            
            # Initialize tracking variables
            results = []
            processed_images = []
            error_count = 0
            memory_warnings = 0

            # In-memory cache: stores compressed PNG bytes for vis_mask / overlay / original
            # for every image so composite saving never needs to re-run ML inference.
            # vis_cache[filename] = {'original': bytes, 'masks': {model: bytes}, 'overlays': {model: bytes}}
            vis_cache = {}
            
            # Create progress dialog
            progress = QProgressDialog("Processing images...", "Cancel", 0, total_images, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            print(f"Starting processing of {total_images} images...")
            print(f"Batch size: {batch_size}, Display limit: {max_display}")
            
            # Process in batches
            for batch_start in range(0, total_images, batch_size):
                if progress.wasCanceled():
                    break
                    
                batch_end = min(batch_start + batch_size, total_images)
                batch_files = image_files[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_images + batch_size - 1) // batch_size
                
                print(f"Processing batch {batch_num}/{total_batches}: images {batch_start+1}-{batch_end}")
                
                # Check memory before batch
                memory_mb = self.analyzer.get_memory_usage_mb()
                if memory_mb > MEMORY_WARNING_THRESHOLD:
                    memory_warnings += 1
                    print(f"Memory warning: {memory_mb:.1f} MB")
                    self.analyzer.force_cleanup()
                
                batch_results = []
                batch_processed_images = []
                
                for i, filename in enumerate(batch_files):
                    if progress.wasCanceled():
                        break
                        
                    current_progress = batch_start + i
                    progress.setValue(current_progress)
                    progress.setLabelText(
                        f"Batch {batch_num}/{total_batches}: {filename} "
                        f"({current_progress + 1}/{total_images})"
                    )
                    QApplication.processEvents()
                    
                    try:
                        image_path = os.path.join(folder_path, filename)
                        
                        # Process image
                        isolated_image = self.analyzer.isolate_object(image_path)
                        if isolated_image is None:
                            print(f"Could not isolate object in: {filename}")
                            continue
                            
                        # Get classifications
                        phenotype_classifications = self.analyzer.classify_image(isolated_image)
                        orientation_classifications = self.analyzer.classify_orientation(isolated_image)
                        
                        # Clear GPU cache after classification
                        self.analyzer.clear_gpu_cache()
                        
                        # Create result dictionary
                        result = {
                            'Image': filename,
                            'Primary_Phenotype': phenotype_classifications[0][0],
                            'Primary_Phenotype_Confidence': round(phenotype_classifications[0][1], 4),
                            'Primary_Orientation': orientation_classifications[0][0],
                            'Primary_Orientation_Confidence': round(orientation_classifications[0][1], 4),
                        }
                        
                        # Add additional classifications
                        for j, (class_name, confidence) in enumerate(phenotype_classifications[1:], 1):
                            result[f'Phenotype_{j+1}'] = class_name
                            result[f'Phenotype_Confidence_{j+1}'] = round(confidence, 4)
    
                        for j, (orientation, confidence) in enumerate(orientation_classifications[1:], 1):
                            result[f'Orientation_{j+1}'] = orientation
                            result[f'Orientation_Confidence_{j+1}'] = round(confidence, 4)
                        
                        primary_orientation = orientation_classifications[0][0]
                        
                        # Prepare display data only if within display limit
                        store_display_data = (len(processed_images) + len(batch_processed_images)) < max_display
                        image_display_data = {
                            'original': None,
                            'dimension_masks': {},
                            'overlays': {}
                        } if store_display_data else None
                        
                        # Process segmentation models
                        for model_name in self.analyzer.segmentation_models.keys():
                            self.analyzer.current_segmentation = model_name

                            # # Skip pericardium segmentation for Ventral orientation
                            # if model_name == 'Pericardium' and primary_orientation == 'Ventral':
                            #     # Set skipped values for pericardium
                            #     skip_result = {
                            #         f'{model_name}_Area': -1,
                            #         f'{model_name}_Length': -1,
                            #         f'{model_name}_Width': -1,
                            #         f'{model_name}_Ratio': -1,
                            #         f'{model_name}_Box_Length': -1,
                            #         f'{model_name}_Box_Width': -1,
                            #         f'{model_name}_Box_Ratio': -1,
                            #         f'{model_name}_Status': 'Skipped - Ventral',
                            #         f'{model_name}_Percentage': -1,
                            #         f'{model_name}_Area_Status': 'Skipped',
                            #         f'{model_name}_Length_Status': 'Skipped',
                            #         f'{model_name}_Width_Status': 'Skipped',
                            #         f'{model_name}_Used_Orientation': primary_orientation
                            #     }
                            #     result.update(skip_result)
                            #     continue

                            try:
                                mask, original_image, length, width, ratio, box_length, box_width, vis_mask = self.analyzer.segment_image(image_path)
                                area = self.analyzer.calculate_area(mask)
                                
                                # Handle different measurement types for different models
                                if model_name == 'Pericardium':
                                    # Pericardium uses area, box length, and box width measurements
                                    status, percentage, area_status, length_status, width_status = self.analyzer.check_anomaly(
                                        box_length, box_width, model_name, primary_orientation, area=area)
                                    
                                    # Calculate box ratio for Excel output
                                    box_ratio = box_length / box_width if box_width != 0 else 0
                                    
                                    # Store measurements for pericardium with additional details
                                    result.update({
                                        f'{model_name}_Area': round(area, 4),
                                        f'{model_name}_Length': round(length, 4),
                                        f'{model_name}_Width': round(width, 4),
                                        f'{model_name}_Ratio': round(ratio, 4),
                                        f'{model_name}_Box_Length': round(box_length, 4),
                                        f'{model_name}_Box_Width': round(box_width, 4),
                                        f'{model_name}_Box_Ratio': round(box_ratio, 4),
                                        f'{model_name}_Status': status,
                                        f'{model_name}_Percentage': round(percentage, 2),
                                        f'{model_name}_Area_Status': area_status,
                                        f'{model_name}_Length_Status': length_status,
                                        f'{model_name}_Width_Status': width_status,
                                        f'{model_name}_Used_Orientation': primary_orientation
                                    })
                                    
                                elif model_name == 'Head':
                                    # Head uses area, length, and width measurements
                                    status, percentage, area_status, length_status, width_status = self.analyzer.check_anomaly(
                                        box_length, box_width, model_name, primary_orientation, area=area, length=length, width=width)
                                    
                                    result.update({
                                        f'{model_name}_Area': round(area, 4),
                                        f'{model_name}_Length': round(length, 4),
                                        f'{model_name}_Width': round(width, 4),
                                        f'{model_name}_Ratio': round(ratio, 4),
                                        f'{model_name}_Box_Length': round(box_length, 4),
                                        f'{model_name}_Box_Width': round(box_width, 4),
                                        f'{model_name}_Status': status,
                                        f'{model_name}_Percentage': round(percentage, 2),
                                        f'{model_name}_Area_Status': area_status,
                                        f'{model_name}_Length_Status': length_status,
                                        f'{model_name}_Width_Status': width_status,
                                        f'{model_name}_Used_Orientation': primary_orientation
                                    })
                                
                                elif model_name == 'Yolksac_ext':
                                    # Yolksac_ext uses area, box length, and box width measurements
                                    status, percentage, area_status, length_status, width_status = self.analyzer.check_anomaly(
                                        box_length, box_width, model_name, primary_orientation, area=area)
                                    
                                    # Calculate box ratio for Excel output
                                    box_ratio = box_length / box_width if box_width != 0 else 0
                                    
                                    # Store measurements for yolksac_ext with additional details
                                    result.update({
                                        f'{model_name}_Area': round(area, 4),
                                        f'{model_name}_Length': round(length, 4),
                                        f'{model_name}_Width': round(width, 4),
                                        f'{model_name}_Ratio': round(ratio, 4),
                                        f'{model_name}_Box_Length': round(box_length, 4),
                                        f'{model_name}_Box_Width': round(box_width, 4),
                                        f'{model_name}_Box_Ratio': round(box_ratio, 4),
                                        f'{model_name}_Status': status,
                                        f'{model_name}_Percentage': round(percentage, 2),
                                        f'{model_name}_Area_Status': area_status,
                                        f'{model_name}_Length_Status': length_status,
                                        f'{model_name}_Width_Status': width_status,
                                        f'{model_name}_Used_Orientation': primary_orientation
                                    })
                                
                                elif model_name == 'Yolk_Sac':
                                    # Yolk_Sac uses area, box length, and box width measurements
                                    status, percentage, area_status, length_status, width_status = self.analyzer.check_anomaly(
                                        box_length, box_width, model_name, primary_orientation, area=area)
                                    
                                    # Calculate box ratio for Excel output
                                    box_ratio = box_length / box_width if box_width != 0 else 0
                                    
                                    # Store measurements for yolk_sac with additional details
                                    result.update({
                                        f'{model_name}_Area': round(area, 4),
                                        f'{model_name}_Length': round(length, 4),
                                        f'{model_name}_Width': round(width, 4),
                                        f'{model_name}_Ratio': round(ratio, 4),
                                        f'{model_name}_Box_Length': round(box_length, 4),
                                        f'{model_name}_Box_Width': round(box_width, 4),
                                        f'{model_name}_Box_Ratio': round(box_ratio, 4),
                                        f'{model_name}_Status': status,
                                        f'{model_name}_Percentage': round(percentage, 2),
                                        f'{model_name}_Area_Status': area_status,
                                        f'{model_name}_Length_Status': length_status,
                                        f'{model_name}_Width_Status': width_status,
                                        f'{model_name}_Used_Orientation': primary_orientation
                                    })
                                    
                                else:
                                    # All other models now use area + box length WITH INDIVIDUAL STATUSES
                                    status, percentage, area_status, length_status = self.analyzer.check_anomaly(
                                        box_length, box_width, model_name, primary_orientation, area=area)

                                    result.update({
                                        f'{model_name}_Area': round(area, 4),
                                        f'{model_name}_Length': round(length, 4),
                                        f'{model_name}_Width': round(width, 4),
                                        f'{model_name}_Ratio': round(ratio, 4),
                                        f'{model_name}_Box_Length': round(box_length, 4),
                                        f'{model_name}_Box_Width': round(box_width, 4),
                                        f'{model_name}_Control_Measurement': round(box_length, 4),
                                        f'{model_name}_Status': status,
                                        f'{model_name}_Percentage': round(percentage, 2),
                                        f'{model_name}_Area_Status': area_status,
                                        f'{model_name}_Box_Length_Status': length_status,
                                        f'{model_name}_Used_Orientation': primary_orientation
                                    })

                                    # For Whole Larva: add curvature to result dict
                                    if model_name == 'Whole Larva':
                                        result['Whole Larva_Curvature'] = getattr(
                                            self.analyzer, '_last_curvature_label', 'N/A'
                                        )
                                
                                # Compute overlay once (used for both display data and temp file)
                                overlay_arr = self.analyzer.create_overlay(original_image, mask)

                                # Store display data only if within display limit
                                if store_display_data and image_display_data is not None:
                                    if image_display_data['original'] is None:
                                        image_display_data['original'] = original_image
                                    image_display_data['dimension_masks'][model_name] = vis_mask
                                    image_display_data['overlays'][model_name] = overlay_arr

                                # Cache compressed PNG bytes in memory for ALL images so
                                # composite saving never needs to re-run ML inference.
                                try:
                                    if filename not in vis_cache:
                                        vis_cache[filename] = {'original': None, 'masks': {}, 'overlays': {}}
                                    cache_entry = vis_cache[filename]
                                    _, vis_buf = cv2.imencode('.png', vis_mask)
                                    cache_entry['masks'][model_name] = vis_buf.tobytes()
                                    _, ovl_buf = cv2.imencode('.png', overlay_arr)
                                    cache_entry['overlays'][model_name] = ovl_buf.tobytes()
                                    if cache_entry['original'] is None:
                                        orig_np = (np.array(original_image)
                                                   if isinstance(original_image, Image.Image)
                                                   else original_image)
                                        if len(orig_np.shape) == 2:
                                            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
                                        _, orig_buf = cv2.imencode('.png', orig_np)
                                        cache_entry['original'] = orig_buf.tobytes()
                                except Exception:
                                    pass  # Non-critical; fallback re-segmentation still available

                                # Clear memory immediately
                                del mask, vis_mask, overlay_arr
                                self.analyzer.clear_gpu_cache()
                                
                            except Exception as e:
                                print(f"Segmentation error for {model_name} in {filename}: {str(e)}")
                                error_count += 1
                                # Set error values based on model type
                                base_error_result = {
                                    f'{model_name}_Area': -1,
                                    f'{model_name}_Length': -1,
                                    f'{model_name}_Width': -1,
                                    f'{model_name}_Ratio': -1,
                                    f'{model_name}_Box_Length': -1,
                                    f'{model_name}_Box_Width': -1,
                                    f'{model_name}_Control_Measurement': -1,
                                    f'{model_name}_Status': 'Error',
                                    f'{model_name}_Percentage': -1,
                                    f'{model_name}_Used_Orientation': primary_orientation
                                }
                                
                                # Add special fields for specific models
                                if model_name in ['Pericardium', 'Yolk_Sac', 'Yolksac_ext']:
                                    base_error_result[f'{model_name}_Box_Ratio'] = -1
                                    base_error_result[f'{model_name}_Area_Status'] = 'Error'
                                    base_error_result[f'{model_name}_Length_Status'] = 'Error'
                                    base_error_result[f'{model_name}_Width_Status'] = 'Error'
                                elif model_name == 'Head':
                                    base_error_result[f'{model_name}_Area_Status'] = 'Error'
                                    base_error_result[f'{model_name}_Length_Status'] = 'Error'
                                    base_error_result[f'{model_name}_Width_Status'] = 'Error'
                                else:
                                    # All other models get area and box length statuses
                                    base_error_result[f'{model_name}_Area_Status'] = 'Error'
                                    base_error_result[f'{model_name}_Box_Length_Status'] = 'Error'
                                    if model_name == 'Whole Larva':
                                        base_error_result['Whole Larva_Curvature'] = 'Error'

                                result.update(base_error_result)
                        
                        batch_results.append(result)
                        
                        # Add to display data if storing
                        if store_display_data and image_display_data is not None:
                            batch_processed_images.append(image_display_data)
                        
                        # Clear variables
                        del isolated_image, phenotype_classifications, orientation_classifications
                        
                        # Aggressive memory check for large batches
                        if is_large_batch and (current_progress + 1) % 10 == 0:
                            memory_mb = self.analyzer.get_memory_usage_mb()
                            if memory_mb > MEMORY_CRITICAL_THRESHOLD:
                                print(f"Critical memory usage: {memory_mb:.1f} MB - forcing cleanup")
                                self.analyzer.force_cleanup()
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        error_count += 1
                        continue
                
                # Add batch to results
                results.extend(batch_results)
                processed_images.extend(batch_processed_images)
                
                # Save checkpoint after each batch
                if results:
                    try:
                        df = pd.DataFrame(results)
                        df = df.iloc[sorted(range(len(df)), key=lambda i: _natural_sort_key(df['Image'].iloc[i]))]
                        checkpoint_file = os.path.join(folder_path, f"checkpoint_batch_{batch_num}.xlsx")
                        df.to_excel(checkpoint_file, index=False)
                        print(f"Checkpoint saved: {len(results)} results processed")
                    except Exception as e:
                        print(f"Error saving checkpoint: {str(e)}")
                
                # Force cleanup after each batch
                self.analyzer.force_cleanup()
                
                # Progress update
                memory_mb = self.analyzer.get_memory_usage_mb()
                print(f"Batch {batch_num} complete. Memory: {memory_mb:.1f} MB, Errors: {error_count}")
            
            progress.close()
            
            # Save final results
            if results:
                try:
                    df = pd.DataFrame(results)
                    df = df.iloc[sorted(range(len(df)), key=lambda i: _natural_sort_key(df['Image'].iloc[i]))]
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    final_file = os.path.join(folder_path, f"analysis_results_{timestamp}.xlsx")
                    df.to_excel(final_file, index=False)
                    
                    # Clean up checkpoint files
                    for i in range(1, ((total_images + batch_size - 1) // batch_size) + 1):
                        checkpoint_file = os.path.join(folder_path, f"checkpoint_batch_{i}.xlsx")
                        if os.path.exists(checkpoint_file):
                            os.remove(checkpoint_file)
                    
                    # Create anomaly folders and save composite images
                    self.create_anomaly_folders(folder_path, results, processed_images,
                                                self.include_phenotype_checkbox.isChecked(),
                                                self.include_orientation_checkbox.isChecked(),
                                                vis_cache)
                    
                    # Setup GUI
                    self.results = results
                    self.processed_images = processed_images
                    self.current_folder = folder_path
                    self.current_index = 0 if processed_images else -1
                    
                    if processed_images:
                        self.display_current_image()
                        self.update_navigation_buttons()
                    
                    # Success message
                    success_msg = f"✅ Processing Complete!\n\n"
                    success_msg += f"📊 Processed: {len(results)} images\n"
                    success_msg += f"❌ Errors: {error_count} images\n"
                    success_msg += f"💾 Results: {os.path.basename(final_file)}\n"
                    success_msg += f"🖼️ GUI Display: {len(processed_images)} images\n"
                    success_msg += f"📁 Anomaly folders created with individual, combination, and normal classifications\n"
                    if memory_warnings > 0:
                        success_msg += f"⚠️ Memory warnings: {memory_warnings}\n"
                    
                    QMessageBox.information(self, "Processing Complete", success_msg)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Could not save results: {str(e)}")
            else:
                QMessageBox.warning(self, "No Results", "No images were processed successfully.")
                
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Processing failed: {str(e)}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
    
    # Also add this helper method to the DanioAnalyzerGUI class for better navigation
    def update_navigation_info(self):
        """Update navigation information display"""
        if self.current_index >= 0 and self.results:
            total_results = len(self.results)
            total_images = len(self.processed_images)
            
            if total_images < total_results:
                nav_info = f"Viewing image {self.current_index + 1} of {total_images} displayed (Total processed: {total_results})"
            else:
                nav_info = f"Image {self.current_index + 1} of {total_results}"
                
            # You can add this to your window title or status bar
            self.setWindowTitle(f"DanioV16 Analyzer - {nav_info}")
    
    def update_navigation_buttons(self):
        """Update navigation buttons and info"""
        if hasattr(self, 'prev_button') and hasattr(self, 'next_button'):
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < len(self.processed_images) - 1)
        
        # Update navigation info
        self.update_navigation_info()
    
    # Add memory monitoring method to ImageAnalyzer class
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0  # Return 0 if psutil not available
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Optional: Add this method to check if you should reduce display images
    def should_reduce_display_images(self, current_count, total_files):
        """Determine if we should reduce displayed images based on memory"""
        memory_mb = self.get_memory_usage_mb()
        
        # If memory usage is high, reduce display images
        if memory_mb > 3000:  # 3GB threshold
            return True
        
        # If processing many files, limit display
        if total_files > 200 and current_count > 50:
            return True
            
        return False

####################################################################################

    def display_current_image(self):
        if self.current_index < 0 or not self.processed_images:
            return
        
        current = self.processed_images[self.current_index]
        result = self.results[self.current_index]
        
        # Get current model measurements including box dimensions
        current_model = self.analyzer.current_segmentation
        area = result.get(f'{current_model}_Area', 0)
        length = result.get(f'{current_model}_Length', 0)
        width = result.get(f'{current_model}_Width', 0)
        ratio = result.get(f'{current_model}_Ratio', 0)
        box_length = result.get(f'{current_model}_Box_Length', 0)
        box_width = result.get(f'{current_model}_Box_Width', 0)
        control_measurement = result.get(f'{current_model}_Control_Measurement', 0)
        status = result.get(f'{current_model}_Status', 'Unknown')
        percentage = result.get(f'{current_model}_Percentage', 0)
        used_orientation = result.get(f'{current_model}_Used_Orientation', 'Unknown')
        
        # Build classification text with both phenotype and orientation
        image_name = result.get('Image', '')
        classification_text = f"Image: {image_name}\n\nPhenotype Classifications:\n"
        classification_text += f"1. {result['Primary_Phenotype']} ({result['Primary_Phenotype_Confidence']:.2f})\n"
        
        i = 2
        while f'Phenotype_{i}' in result:
            classification_text += f"{i}. {result[f'Phenotype_{i}']} ({result[f'Phenotype_Confidence_{i}']:.2f})\n"
            i += 1
        
        classification_text += "\nOrientation Classifications:\n"
        classification_text += f"1. {result['Primary_Orientation']} ({result['Primary_Orientation_Confidence']:.2f})\n"
        
        i = 2
        while f'Orientation_{i}' in result:
            classification_text += f"{i}. {result[f'Orientation_{i}']} ({result[f'Orientation_Confidence_{i}']:.2f})\n"
            i += 1
        
        # Build measurement text based on model type
        # Build measurement text based on model type
        if current_model == 'Pericardium':
            # Special display for pericardium with triple measurements
            box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
            area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
            length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
            width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
            
            measurement_text = (
                f"{current_model} Measurements (Using {used_orientation} controls):\n"
                f"Area: {area:.2f} mm² (Status: {area_status})\n"
                f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                f"Overall Status: {status} | Percentage: {percentage:.1f}%"
            )
        elif current_model == 'Head':
            # Special display for head with triple measurements
            area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
            length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
            width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
            
            measurement_text = (
                f"{current_model} Measurements (Using {used_orientation} controls):\n"
                f"Area: {area:.2f} mm² (Status: {area_status})\n"
                f"Length: {length:.2f} mm (Status: {length_status})\n"
                f"Width: {width:.2f} mm (Status: {width_status})\n"
                f"Box Length: {box_length:.2f} mm | Box Width: {box_width:.2f} mm\n"
                f"Overall Status: {status} | Percentage: {percentage:.1f}%"
            )
            
        elif current_model == 'Yolksac_ext':
            # Special display for yolksac_ext with triple measurements
            box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
            area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
            length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
            width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
            
            measurement_text = (
                f"{current_model} Measurements (Using {used_orientation} controls):\n"
                f"Area: {area:.2f} mm² (Status: {area_status})\n"
                f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                f"Overall Status: {status} | Percentage: {percentage:.1f}%"
            )
        
        elif current_model == 'Yolk_Sac':
            # Special display for yolk_sac with triple measurements
            box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
            area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
            length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
            width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
            
            measurement_text = (
                f"{current_model} Measurements (Using {used_orientation} controls):\n"
                f"Area: {area:.2f} mm² (Status: {area_status})\n"
                f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                f"Overall Status: {status} | Percentage: {percentage:.1f}%"
            )
        else:
            # Standard display for other models WITH INDIVIDUAL STATUSES
            area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
            length_status = result.get(f'{current_model}_Box_Length_Status', 'Unknown')

            measurement_text = (
                f"{current_model} Measurements (Using {used_orientation} controls):\n"
                f"Area: {area:.2f} mm² (Status: {area_status})\n"
                f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                f"Box Width: {box_width:.2f} mm (for reference only)\n"
                f"Overall Status: {status} | Percentage: {percentage:.1f}%"
            )

            # Append curvature line for Whole Larva
            if current_model == 'Whole Larva':
                curvature = result.get('Whole Larva_Curvature', 'N/A')
                if curvature not in ('N/A', 'Error', ''):
                    measurement_text += f"\nCurvature: {curvature}"
        
        self.measurement_label.setText(f"{classification_text}\n{measurement_text}")
        
        # Display images with dimension lines
        self.display_image(current['original'], self.original_label)
        self.display_image(current['dimension_masks'][current_model], self.mask_label)
        self.display_image(current['overlays'][current_model], self.overlay_label)
    
    # 7. Replace the change_segmentation_model method
    
    def change_segmentation_model(self, model_name):
        # Update button states
        for btn in self.segmentation_buttons.values():
            btn.setChecked(btn.text() == model_name)
        
        try:
            self.analyzer.current_segmentation = model_name
            if self.current_index >= 0:
                image_path = os.path.join(self.current_folder, self.results[self.current_index]['Image'])
                mask, original_image, length, width, ratio, box_length, box_width, vis_mask = self.analyzer.segment_image(image_path)
                
                # Calculate area for current segmentation
                area = self.analyzer.calculate_area(mask)
                
                # Update measurement label with multiple classifications
                result = self.results[self.current_index]
                
                # Get current model measurements including box dimensions
                current_model = self.analyzer.current_segmentation
                model_area = result.get(f'{current_model}_Area', area)
                model_length = result.get(f'{current_model}_Length', length)
                model_width = result.get(f'{current_model}_Width', width)
                model_ratio = result.get(f'{current_model}_Ratio', ratio)
                model_box_length = result.get(f'{current_model}_Box_Length', box_length)
                model_box_width = result.get(f'{current_model}_Box_Width', box_width)
                control_measurement = result.get(f'{current_model}_Control_Measurement', 0)
                status = result.get(f'{current_model}_Status', 'Unknown')
                percentage = result.get(f'{current_model}_Percentage', 0)
                used_orientation = result.get(f'{current_model}_Used_Orientation', 'Unknown')
                
                # Build classification text
                image_name = result.get('Image', '')
                classification_text = f"Image: {image_name}\n\nPhenotype Classifications:\n"
                classification_text += f"1. {result['Primary_Phenotype']} ({result['Primary_Phenotype_Confidence']:.2f})\n"
                
                i = 2
                while f'Phenotype_{i}' in result:
                    classification_text += f"{i}. {result[f'Phenotype_{i}']} ({result[f'Phenotype_Confidence_{i}']:.2f})\n"
                    i += 1
                
                classification_text += "\nOrientation Classifications:\n"
                classification_text += f"1. {result['Primary_Orientation']} ({result['Primary_Orientation_Confidence']:.2f})\n"
                
                i = 2
                while f'Orientation_{i}' in result:
                    classification_text += f"{i}. {result[f'Orientation_{i}']} ({result[f'Orientation_Confidence_{i}']:.2f})\n"
                    i += 1
                
                # Build measurement text based on model type
                if current_model == 'Pericardium':
                    # Special display for pericardium with triple measurements
                    box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
                    area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
                    length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
                    width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
                    
                    measurement_text = (
                        f"{current_model} Measurements (Using {used_orientation} controls):\n"
                        f"Area: {area:.2f} mm² (Status: {area_status})\n"
                        f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                        f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                        f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                        f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                        f"Overall Status: {status} | Percentage: {percentage:.1f}%"
                    )
                elif current_model == 'Head':
                    # Special display for head with triple measurements
                    area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
                    length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
                    width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
                    
                    measurement_text = (
                        f"{current_model} Measurements (Using {used_orientation} controls):\n"
                        f"Area: {area:.2f} mm² (Status: {area_status})\n"
                        f"Length: {length:.2f} mm (Status: {length_status})\n"
                        f"Width: {width:.2f} mm (Status: {width_status})\n"
                        f"Box Length: {box_length:.2f} mm | Box Width: {box_width:.2f} mm\n"
                        f"Overall Status: {status} | Percentage: {percentage:.1f}%"
                    )
                    
                elif current_model == 'Yolksac_ext':
                    # Special display for yolksac_ext with triple measurements
                    box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
                    area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
                    length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
                    width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
                    
                    measurement_text = (
                        f"{current_model} Measurements (Using {used_orientation} controls):\n"
                        f"Area: {area:.2f} mm² (Status: {area_status})\n"
                        f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                        f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                        f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                        f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                        f"Overall Status: {status} | Percentage: {percentage:.1f}%"
                    )    
                
                elif current_model == 'Yolk_Sac':
                    # Special display for yolk_sac with triple measurements
                    box_ratio = result.get(f'{current_model}_Box_Ratio', 0)
                    area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
                    length_status = result.get(f'{current_model}_Length_Status', 'Unknown')
                    width_status = result.get(f'{current_model}_Width_Status', 'Unknown')
                    
                    measurement_text = (
                        f"{current_model} Measurements (Using {used_orientation} controls):\n"
                        f"Area: {area:.2f} mm² (Status: {area_status})\n"
                        f"Length: {length:.2f} mm | Width: {width:.2f} mm | Ratio: {ratio:.2f}\n"
                        f"Box Length: {box_length:.2f} mm (Status: {length_status})\n"
                        f"Box Width: {box_width:.2f} mm (Status: {width_status})\n"
                        f"Box Ratio: {box_ratio:.2f} (for reference only)\n"
                        f"Overall Status: {status} | Percentage: {percentage:.1f}%"
                    )
                else:
                    # Standard display for other models WITH INDIVIDUAL STATUSES
                    area_status = result.get(f'{current_model}_Area_Status', 'Unknown')
                    length_status = result.get(f'{current_model}_Box_Length_Status', 'Unknown')

                    measurement_text = (
                        f"{model_name} Measurements (Using {used_orientation} controls):\n"
                        f"Area: {model_area:.2f} mm² (Status: {area_status})\n"
                        f"Length: {model_length:.2f} mm | Width: {model_width:.2f} mm | Ratio: {model_ratio:.2f}\n"
                        f"Box Length: {model_box_length:.2f} mm (Status: {length_status})\n"
                        f"Box Width: {model_box_width:.2f} mm (for reference only)\n"
                        f"Overall Status: {status} | Percentage: {percentage:.1f}%"
                    )

                    # Append curvature line for Whole Larva
                    if current_model == 'Whole Larva':
                        curvature = result.get('Whole Larva_Curvature', 'N/A')
                        if curvature not in ('N/A', 'Error', ''):
                            measurement_text += f"\nCurvature: {curvature}"

                self.measurement_label.setText(f"{classification_text}\n{measurement_text}")
                
                # Display images
                self.display_image(original_image, self.original_label)
                self.display_image(vis_mask, self.mask_label)
                overlay = self.analyzer.create_overlay(original_image, mask)
                self.display_image(overlay, self.overlay_label)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update segmentation: {str(e)}")

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
            self.update_navigation_buttons()

    def show_next(self):
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.display_current_image()
            self.update_navigation_buttons()
            
    def update_navigation_info(self):
        """Update navigation information display"""
        if self.current_index >= 0 and self.results:
            total_results = len(self.results)
            total_images = len(self.processed_images)
            
            if total_images < total_results:
                nav_info = f"Viewing image {self.current_index + 1} of {total_images} displayed (Total processed: {total_results})"
            else:
                nav_info = f"Image {self.current_index + 1} of {total_results}"
                
            # Update window title with navigation info
            self.setWindowTitle(f"DanioV16 Analyzer - {nav_info}")
    
    def update_navigation_buttons(self):
        """Enhanced navigation button updates"""
        if hasattr(self, 'prev_button') and hasattr(self, 'next_button'):
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < len(self.processed_images) - 1)
        
        # Update navigation info
        self.update_navigation_info()
    
    def display_batch_info(self):
        """Display information about current batch processing"""
        if self.results and self.processed_images:
            total_processed = len(self.results)
            total_displayable = len(self.processed_images)
            
            if total_displayable < total_processed:
                info_text = f"Displaying {total_displayable} of {total_processed} processed images"
                info_text += f"\n(Memory optimization: showing first {total_displayable} images)"
                info_text += f"\nAll results saved to Excel file"
            else:
                info_text = f"All {total_processed} images available for viewing"
            
            # You can add this to a status label if you have one
            print(info_text)  # For now, print to console

    def display_image(self, image, label):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())