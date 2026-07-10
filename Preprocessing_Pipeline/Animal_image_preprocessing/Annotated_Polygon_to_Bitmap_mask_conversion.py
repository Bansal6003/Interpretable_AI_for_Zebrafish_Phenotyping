import cv2
import numpy as np
import os

def yolo_polygon_to_mask(image_path, polygon_points, mask_output_path):
    """
    Converts YOLOv9 polygon points to a segmentation mask for a given image.
    
    Parameters:
    - image_path: path to the image
    - polygon_points: list of (x, y) tuples in YOLOv9 normalized format [(x1, y1), (x2, y2), ...]
    - mask_output_path: where to save the generated mask
    """
    
    # Load the image to get its dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Create a blank mask with the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert the YOLO normalized polygon points to absolute pixel coordinates
    absolute_polygon_points = []
    for point in polygon_points:
        x_abs = int(point[0] * width)
        y_abs = int(point[1] * height)
        absolute_polygon_points.append([x_abs, y_abs])

    # Convert the list to a numpy array
    polygon_array = np.array([absolute_polygon_points], dtype=np.int32)

    # Draw and fill the polygon on the mask
    cv2.fillPoly(mask, polygon_array, 255)

    # Save the generated mask
    cv2.imwrite(mask_output_path, mask)

def read_polygon_points_from_txt(txt_file):
    """
    Reads YOLOv9 polygon points from a .txt file.
    
    Parameters:
    - txt_file: path to the .txt file containing polygon points
    
    Returns:
    - A list of polygon points in YOLOv9 normalized format [(x1, y1), (x2, y2), ...]
    """
    polygon_points_list = []
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            data = line.strip().split()
            # Ignore the class_id, take the rest as the polygon points
            polygon_points = [(float(data[i]), float(data[i + 1])) for i in range(1, len(data), 2)]
            polygon_points_list.append(polygon_points)
    
    return polygon_points_list

def process_images(image_folder, txt_folder, output_folder):
    """
    Processes multiple images, reads corresponding polygon points from .txt files,
    and converts the polygons to segmentation masks.
    
    Parameters:
    - image_folder: folder containing the images
    - txt_folder: folder containing the .txt files with polygon points
    - output_folder: folder where the segmentation masks will be saved
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the image folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            txt_file = os.path.join(txt_folder, os.path.splitext(image_file)[0] + '.txt')
            
            if os.path.exists(txt_file):
                # Read polygon points from the corresponding .txt file
                all_polygons = read_polygon_points_from_txt(txt_file)
                
                # Create mask output path using the original image file name (without extension)
                mask_output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.png')
                
                for polygon_points in all_polygons:
                    yolo_polygon_to_mask(image_path, polygon_points, mask_output_path)
                    print(f"Mask saved for {image_file} at {mask_output_path}")

# Example usage

# Folder containing the images
image_folder = r"C:\Users\Pushkar Bansal\Downloads\yolk_sac_ext.v2i.yolov11\train\images"

# Folder containing the .txt files with polygon annotations
txt_folder = r"C:\Users\Pushkar Bansal\Downloads\yolk_sac_ext.v2i.yolov11\train\labels"

# Folder where the segmentation masks will be savedr"C:\Users\Pushkar Bansal\Downloads\Fish Pericardium.v1i.yolov11\train_masks"
output_folder =  r"C:\Users\Pushkar Bansal\Downloads\yolk_sac_ext.v2i.yolov11\train_masks"
# Process the images and save the masks
process_images(image_folder, txt_folder, output_folder)

