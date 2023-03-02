import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy as np
# to detect the face of the human
from retinaface import RetinaFace

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = image_path/255
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor



# Obtain plate image and its coordinates from an image
webcam= cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img2.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Image saved!")     
            break
        elif key == ord('q'):
            cv2.imwrite(filename='saved_img4.jpg', img=frame)
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break 
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
        
resp = RetinaFace.detect_faces("saved_img4.jpg")
try:
    for i in range(len(resp)):
        globals()['a_'+str(i+1)]=resp["face_"+str(i+1)]["facial_area"]
        frame[globals()['a_'+str(i+1)][1]:globals()['a_'+str(i+1)][3],globals()['a_'+str(i+1)][0]:globals()['a_'+str(i+1)][2]]=cv2.medianBlur(frame[globals()['a_'+str(i+1)][1]:globals()['a_'+str(i+1)][3], globals()['a_'+str(i+1)][0]:globals()['a_'+str(i+1)][2]], 101)
except:
    pass


try:
    LpImg,cor = get_plate(frame)
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    pts = np.array(pts, np.int32)
    frame = cv2.rectangle(frame, pts[1], pts[3], (50,50,50), -1)
    frame = cv2.rectangle(frame, pts[0], pts[2], (50,50,50), -1)
    
except: 
    print("license plate couldnt detected.")


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cv2.imwrite(filename='saved_img.jpg', img=frame)