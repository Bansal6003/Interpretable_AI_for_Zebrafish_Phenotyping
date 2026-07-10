import torch
from torch import nn
from torchvision import transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, QScrollArea
from PyQt5.QtCore import Qt
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import timm
import traceback
import sys
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main Application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Danio AI: ')
        self.setGeometry(100, 100, 600, 400)

        self.classification_button = QPushButton('Object Isolation', self)
        self.classification_button.clicked.connect(self.open_isolator)
        self.classification_button.setGeometry(200, 100, 200, 50)

    def open_isolator(self):
        # model_path = YOLO(r"D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_3dpf_models_real_deal\whole_larva_v2\weights\last.pt")
        # model_path = YOLO(r".\Best Models_regularly updated\for_dark_BG_isolation\last.pt")
        # model_path = YOLO(r".\Best Models_regularly updated\for_dark_BG_isolation\last.pt")
        model_path = YOLO(r"C:\Users\Pushkar Bansal\runs\segment\train13\weights\last.pt")
        self.Isolator_window = Isolation_Window(model_path)
        self.Isolator_window.show()

# Classification Window
class Isolation_Window(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle('Object isolation')
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        
        # Load the YOLO model
        try:
            # self.model = YOLO(r"D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_3dpf_models_real_deal\whole_larva_v2\weights\last.pt")
            # self.model = YOLO(r".\Best Models_regularly updated\for_dark_BG_isolation\last.pt")
            self.model = YOLO(r"C:\Users\Pushkar Bansal\runs\segment\train13\weights\last.pt")
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load YOLO model: {e}")
            sys.exit(1)

    def initUI(self):
        main_layout = QVBoxLayout()
        
        # Upload button
        self.button = QPushButton("Upload Folder")
        self.button.clicked.connect(self.upload_folder)
        main_layout.addWidget(self.button)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Original Test Images')
        if not folder_path:
            print("No folder selected.")
            return

        print(f"Selected folder: {folder_path}")

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1

        isolated_objects_directory = os.path.join(folder_path, "isolated_objects")
        
        for directory in [isolated_objects_directory]:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        print(f"Found {len(image_files)} image files.")

        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Failed to read image: {image_path}")
                continue

            self.image_paths.append(image_path)

            try:
                # Process the image
                results = self.model(image_path, show=False, save=False)
                predicted_img = results[0].plot()
                print(f"YOLO processing completed for {filename}")

                # Extract and save isolated objects with square bounding boxes
                self.extract_objects_square(original_image, results[0], isolated_objects_directory, filename)

                # Resize the image for display
                resized_img = cv2.resize(predicted_img, (800, 600))
                self.processed_images.append(resized_img)
                print(f"Added processed image for {filename}. Total processed: {len(self.processed_images)}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print(traceback.format_exc())

        print(f"Total processed images: {len(self.processed_images)}")

        if self.processed_images:
            self.current_index = 0
            QMessageBox.information(self, 'Success', f'Images processed and objects isolated. Total processed: {len(self.processed_images)}')
        else:
            QMessageBox.warning(self, 'Warning', 'No images were processed.')

    def extract_objects_square(self, image, result, output_directory, filename, fixed_square_size=1500, target_size=1920, padding_color=(0, 0, 0)): #original 4000
        """
        Extract objects using fixed square bounding boxes for truly consistent sizing with background removal.
        
        Parameters:
        - image: Original image
        - result: YOLO detection result
        - output_directory: Directory to save isolated objects
        - filename: Original filename
        - fixed_square_size: Fixed size for extraction square (before resizing)
        - target_size: Size of the final output image (default: 224x224)
        - padding_color: Color for padding (default: black)
        """
        print(f"Extracting objects from {filename} with fixed square bounding boxes and background removal")
        print(f"Result type: {type(result)}")
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            print(f"Number of detected objects: {len(boxes)}")
            print(f"Masks available: {masks is not None}")
            
            for i, box in enumerate(boxes):
                try:
                    print(f"Processing box {i+1}")
                    
                    if hasattr(box, 'xyxy'):
                        # Get the coordinates of the bounding box
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        print(f"Original object {i+1} coordinates: ({x1}, {y1}, {x2}, {y2})")
                        
                        # Calculate center of the detected object
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Use FIXED square size for all objects
                        # This ensures diagonal and horizontal larvae of same size appear identical
                        half_size = fixed_square_size // 2
                        square_x1 = max(0, center_x - half_size)
                        square_y1 = max(0, center_y - half_size)
                        square_x2 = min(image.shape[1], center_x + half_size)
                        square_y2 = min(image.shape[0], center_y + half_size)
                        
                        print(f"Fixed square object {i+1} coordinates: ({square_x1}, {square_y1}, {square_x2}, {square_y2})")
                        print(f"Extracted square size: {square_x2-square_x1} x {square_y2-square_y1}")
                        
                        # Extract the square region
                        object_img = image[square_y1:square_y2, square_x1:square_x2].copy()
                        
                        # Apply background removal if masks are available
                        if masks is not None and i < len(masks.data):
                            print(f"Applying background removal for object {i+1}")
                            
                            # Get the mask for this specific object
                            mask = masks.data[i].cpu().numpy()
                            
                            # Resize mask to match original image size if needed
                            if mask.shape != (image.shape[0], image.shape[1]):
                                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            # Extract the mask region corresponding to our square
                            mask_region = mask[square_y1:square_y2, square_x1:square_x2]
                            
                            # Improve mask to preserve more of the larva
                            mask_region = self.improve_mask(mask_region)
                            
                            # Convert mask to 3-channel for multiplication
                            if len(mask_region.shape) == 2:
                                mask_region = np.stack([mask_region] * 3, axis=2)
                            
                            # Apply mask to remove background (multiply image by mask)
                            # Pixels where mask is 0 will become black (background removed)
                            object_img = object_img * mask_region
                            
                            print(f"Background removed for object {i+1}")
                        else:
                            print(f"No mask available for object {i+1}, keeping original extraction")
                        
                        if object_img.size > 0:
                            # Resize to target size while maintaining aspect ratio
                            object_img_resized = self.resize_with_padding(object_img, target_size, padding_color)
                            
                            # Save the extracted object
                            obj_filename = f'object_{os.path.splitext(filename)[0]}_{i}.png'
                            obj_path = os.path.join(output_directory, obj_filename)
                            success = cv2.imwrite(obj_path, object_img_resized)
                            if success:
                                print(f"Saved isolated object: {obj_path} (size: {target_size}x{target_size})")
                            else:
                                print(f"Failed to save isolated object: {obj_path}")
                        else:
                            print(f"Warning: Empty extracted image for object {i+1}")
                    else:
                        print(f"Box {i+1} does not have 'xyxy' attribute")
                except Exception as e:
                    print(f"Error extracting object {i+1}: {e}")
                    print(traceback.format_exc())
        else:
            print("No boxes found in the result")

    def improve_mask(self, mask_region, dilation_kernel_size=300, blur_kernel_size=5, threshold=0.1):
        """
        Improve the segmentation mask to preserve more of the larva.
        
        Parameters:
        - mask_region: The extracted mask region
        - dilation_kernel_size: Size of dilation kernel to expand the mask
        - blur_kernel_size: Size of Gaussian blur kernel for smoother edges
        - threshold: Threshold for the blurred mask (lower = more inclusive)
        
        Returns:
        - Improved mask that preserves more of the larva
        """
        try:
            # Ensure mask is in the right format (0-1 range)
            if mask_region.max() > 1.0:
                mask_region = mask_region / 255.0
            
            # Convert to uint8 for OpenCV operations
            mask_uint8 = (mask_region * 255).astype(np.uint8)
            
            # Method 1: Dilate the mask to expand it slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
            
            # Method 2: Apply Gaussian blur to smooth edges and make transitions less harsh
            blurred_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32), (blur_kernel_size, blur_kernel_size), 0)
            
            # Method 3: Apply a lower threshold to include more pixels
            # This makes the mask more inclusive
            improved_mask = np.where(blurred_mask > (threshold * 255), 1.0, 0.0)
            
            print(f"Mask improvement applied: dilation={dilation_kernel_size}, blur={blur_kernel_size}, threshold={threshold}")
            return improved_mask
            
        except Exception as e:
            print(f"Error improving mask: {e}")
            # Fallback to original mask if improvement fails
            return mask_region

    def resize_with_padding(self, image, target_size, padding_color=(0, 0, 0)):
        """
        Resize image to target size while maintaining aspect ratio and adding padding.
        
        Parameters:
        - image: Input image
        - target_size: Target square size
        - padding_color: Color for padding
        
        Returns:
        - Resized image with padding
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit the image within target_size
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a square canvas with padding color
        canvas = np.full((target_size, target_size, image.shape[2]), padding_color, dtype=image.dtype)
        
        # Calculate position to center the resized image
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return canvas

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()