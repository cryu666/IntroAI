import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    
    with open(data_path) as txt_file:
        h = [int(x) for x in next(txt_file).split()]
        coordinates = []
        for line in txt_file:
            coordinates.append([int(x) for x in line.split()])
        
    
    cap_file = cv2.VideoCapture("data/detect/video.gif")
    while(True):
        
        success, frame = cap_file.read()
        
        pred = []
        for coordinate in coordinates:
            x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
            parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, frame)
            parking_space_image_resize = cv2.resize(parking_space_image, dsize=(36, 16))
            
            parking_space_image_gray = cv2.cvtColor(parking_space_image_resize, cv2.COLOR_BGR2GRAY)
            
            parking_space_image_flatten = parking_space_image_gray.reshape(1, -1)
        
            if clf.classify(parking_space_image_flatten) == 1:
                cv2.rectangle(frame, (x1, y1), (x4, y4), (0, 255, 0), 1)
                pred.append(1)
            else:
                pred.append(0)
                  
                
        cv2.imshow('frame', frame)
        
        pred_file = open("ML_Models_pred.txt", "a+")
        for i in pred:
            pred_file.write(str(i) + " ")
        pred_file.write("\n")
        pred_file.close()
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    cap_file.release()
    cv2.destroyAllWindows()
        
    
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
