import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf


def display_images_from_folders(base_dir, categories, num_images=5):
    
    for i, cat in enumerate(categories):
        cat_path = os.path.join(base_dir, cat, 'CROPED')
        images_files = [f for f in os.listdir(cat_path) if f.endswith(('.bmp'))]
        
        for j, img_file in enumerate(images_files[:num_images]):
            img_path = os.path.join(cat_path, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(image)
            plt.axis('off')  
            plt.title(f"{cat}")  
            
            
base_dir = "Implementations/Datasets/SIPaKMeD"
categories = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", 
              "im_Parabasal", "im_Superficial-Intermediate"]
INPUT_SHAPE = 224

display_images_from_folders(base_dir, categories, num_images=5)
