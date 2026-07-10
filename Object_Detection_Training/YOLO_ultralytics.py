import ultralytics
import os
from ultralytics import YOLO

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # model = ultralytics.YOLO("yolov26n-seg.pt")
    model = YOLO("yolov26-obb.pt")  
  

    # model.train(data=r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\Augmented_Dataset', 
    #             epochs=600, imgsz=256)
    
    # model.train(data=r'C:\Users\Pushkar Bansal\Downloads\48_dpf_new_params_pericardium.v5i.yolov9\data.yaml', 
    #             epochs=1000, imgsz=640)
    
    model.train(data=r'C:\Users\Pushkar Bansal\Downloads\48_dpf_new_params_yolkext_tail.v5i.yolov9\data.yaml', 
                epochs=1000, imgsz=640)
    
    
if __name__ == "__main__":
    main()

#####################################################################################
#                                    Plotting
####################################################################################

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# data = pd.read_csv(r'C:\Users\pkrap\runs\classify\train\results.csv')
# df = pd.DataFrame(data)

# plt.figure()
# sns.lineplot(data = df, x = 'epoch', y = 'train/loss')
# # plt.plot(df['epoch'], df['val/loss'], label='Val loss')
# # plt.title('Training vs. Validation Loss')
# # plt.grid(linestyle='--', linewidth=1)
# # plt.title('Model Performance')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()

# plt.show()

#####################################################################################
#                                    Prediction
####################################################################################

# from ultralytics import YOLO

# model = YOLO(r'C:\Users\pkrap\runs\classify\train\weights\best.pt')

# results = model(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\CNN_image_identification_zebrafish\Zach_larva.tif')

# print(results)

#####################################################################################
#                                Bulk Prediction
####################################################################################

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import os
# from ultralytics import YOLO
# import torch

# # Load your pretrained YOLO model
# model = YOLO(r'C:\Users\pkrap\Desktop\Behavioral genetics\AI_Project_Python_Templates\Working Codes_Dont mess\runs\classify\train\weights\best.pt')

# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     return img

# def classify_images(folder_path):
#     results = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif"):  # Include other formats if needed
#             file_path = os.path.join(folder_path, filename)
#             img = preprocess_image(file_path)
#             # Predict for each image separately
#             predictions = model(img)
#             result_text.insert(tk.END, f"{filename}:\n")
#             for prediction in predictions:
#                 if prediction.boxes:
#                     for box in prediction.boxes:
#                         bbox = box.xyxy  # Extract bounding box coordinates
#                         cls = int(box.cls)  # Extract class index
#                         score = box.conf.item()  # Extract confidence score
#                         result_text.insert(tk.END, f"Class: {cls}, Score: {score}, BBox: {bbox}\n")
#             result_text.insert(tk.END, "\n")  # New line after each file's results
#     return results

# def upload_folder():
#     folder_path = filedialog.askdirectory()
#     if folder_path:
#         classify_images(folder_path)

# # Create the main window
# root = tk.Tk()
# root.title("Image Classification GUI")

# # Create and place the components
# upload_button = tk.Button(root, text="Upload Folder", command=upload_folder)
# upload_button.pack(pady=20)

# result_text = tk.Text(root, wrap='word', height=15, width=50)
# result_text.pack(pady=20)

# # Start the GUI event loop
# root.mainloop()




# r'C:\Users\pkrap\runs\classify\train\weights\best.pt'
