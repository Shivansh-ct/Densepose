#Converting Videos to Frames
import cv2
import numpy as np
import glob
import os

list_files = []
for i in glob.glob('/mnt/scratch1/csy207576/project/Data/dr_vishnu_data/*.MOV'):
 list_files.append(i)

#list_files = ['/home/mtech/csy207576/exp/ex0.mp4']
for dir in list_files:
 print(dir)
 os.system("mkdir /mnt/scratch1/csy207576/project/temp2")
 os.system("mkdir /mnt/scratch1/csy207576/project/temp3")

 #Opens the Video file and extracts all its frames
 cap= cv2.VideoCapture(dir)
 i=0
 while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == False:
    break
  cv2.imwrite('/mnt/scratch1/csy207576/project/temp2/'+str(i)+'.jpg',frame)
  os.system("python3 apply_net.py show configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml /mnt/scratch1/csy207576/densepose/models/model_final_10af0e.pkl /mnt/scratch1/csy207576/project/temp2/"+str(i)+".jpg dp_contour --output /mnt/scratch1/csy207576/project/temp3/"+str(i)+".jpg")
  i+=1

 cap.release()
 #cv2.destroyAllWindows()

 # Getting the frame rate
 video = cv2.VideoCapture(dir);
 # Find OpenCV version
 (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 if int(major_ver)  < 3 :
  fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
 else :
  fps = video.get(cv2.CAP_PROP_FPS)
 video.release()

 # Combining Frames to form a video

 img_array = []
 for filename in glob.glob('/mnt/scratch1/csy207576/project/temp3/*.jpg'):
    img = cv2.imread(filename)
    print(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

 out = cv2.VideoWriter(dir,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
 for i in range(len(img_array)):
    out.write(img_array[i])
 out.release()
 os.system("rm -r /mnt/scratch1/csy207576/project/temp2")
 os.system("rm -r /mnt/scratch1/csy207576/project/temp3")
 #np.save("/content/drive/MyDrivelist.npy",dir)




