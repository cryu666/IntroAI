import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    
    dataset = []
                
    for sub_folder in os.listdir(data_path):
        for filename in os.listdir(os.path.join(data_path, sub_folder)):
            img = cv2.imread(os.path.join(data_path, sub_folder, filename), 0)
            new_img = img.astype(float)
            new_img = cv2.resize(new_img, dsize=(36, 16))
        
            if sub_folder.startswith('non'): 
                label = 0
            else:
                label = 1

            pair = (new_img, label)
            dataset.append(pair)
    
    
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
