import os
import shutil
from pathlib import Path

def extract_images_to_single_folder(source_directory, output_folder_name="all_images"):
    """
    Extract all image files from subdirectories and save them in a single folder.
    
    Args:
        source_directory (str): Path to the directory containing subdirectories with images
        output_folder_name (str): Name of the output folder to create
    """
    
    # Define common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}
    
    # Convert to Path object for easier handling
    source_path = Path(source_directory)
    output_path = source_path / output_folder_name
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Counter for processed files
    processed_files = 0
    duplicate_counter = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        
        # Skip the output directory itself
        if root_path == output_path:
            continue
            
        for file in files:
            file_path = root_path / file
            file_extension = file_path.suffix.lower()
            
            # Check if it's an image file
            if file_extension in image_extensions:
                # Handle duplicate filenames
                output_file_path = output_path / file
                original_name = file_path.stem
                extension = file_path.suffix
                
                # If file already exists, add a number suffix
                counter = 1
                while output_file_path.exists():
                    new_name = f"{original_name}_{counter}{extension}"
                    output_file_path = output_path / new_name
                    counter += 1
                
                try:
                    # Copy the file to the output directory
                    shutil.copy2(file_path, output_file_path)
                    processed_files += 1
                    print(f"Copied: {file_path} -> {output_file_path}")
                    
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")
    
    print(f"\nCompleted! Processed {processed_files} image files.")
    print(f"All images saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = input("Enter the directory path: ").strip()
    
    # Optional: customize output folder name
    output_name = input("Enter output folder name (or press Enter for 'all_images'): ").strip()
    if not output_name:
        output_name = "all_images"
    
    if os.path.exists(directory_path):
        extract_images_to_single_folder(directory_path, output_name)
    else:
        print("Directory not found!")