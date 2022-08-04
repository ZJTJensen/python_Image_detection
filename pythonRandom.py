from PIL import Image
import numpy as np
from random import randint
import imutils 
import cv2
import threading

frame_count = 0
chosenRange = 20
previous_frame = None
prepared_frame = None
old_frame = None
# Change the first field passed into the method to change camera port
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hasSelected = False
noFunctionCalls= True
hog = cv2.HOGDescriptor() 
humanDetected = False
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
displayImages=[]
new_im = Image.open('./imgs/logo.png')
new_im.save('numToDisplay.png', 'PNG')

def decide_IMG():
    global chosenRange
    chosenNum = str(randint(1, int(chosenRange)))
    if(int(chosenNum) > 9):
        displayImages.append('./imgs/'+ chosenNum[0] +'.png')
        displayImages.append('./imgs/'+ chosenNum[1] +'.png')
        fixedImages = []
        if len(displayImages) > 2:
            fixedImages.append(displayImages[0])  
            fixedImages.append(displayImages[1])
        images = [Image.open(x) for x in displayImages]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGBA', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save('numToDisplay.png', 'PNG')
        # new_im.show();
    else: 
        new_im = Image.open('./imgs/'+ chosenNum +'.png')
        new_im.save('numToDisplay.png', 'PNG')
        # new_im.show()

def humanDetector(image, width, height):
    global hog
    global old_frame
    similarity = 1
    image = imutils.resize(image, 
                       width=min(500, image.shape[1])) 
    if old_frame is not None: 
        errorL2 = cv2.norm( image, old_frame, cv2.NORM_L2 )
        similarity = 1 - errorL2 / ( height * width )
    if old_frame is not None and similarity < .9:
        old_frame = image
        (humans, _) = hog.detectMultiScale(image,  
                                        winStride=(5, 5), 
                                        padding=(3, 3), 
                                        scale=1.21)
        if (len(humans)> 0):
            return True
    old_frame = image
    return False

def determineRun():
    global noFunctionCalls
    global hasSelected
    global displayImages
    global humanDetected
    noFunctionCalls = True
    if displayImages == [] and not humanDetected:
        hasSelected = True
        decide_IMG()
    elif displayImages == []:
        humanDetected = False

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
    if((frame_count % 2) == 0):
      prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
      prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)
    if (previous_frame is None):
        previous_frame = prepared_frame
        continue
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if (cv2.contourArea(contour) < 10000):
            if not hasSelected and not humanDetected:
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                humanDetected = humanDetector(
                    img_rgb,
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )
                if noFunctionCalls == True: 
                    noFunctionCalls = False;
                    timmerDelay = threading.Timer(.5, determineRun)
                    timmerDelay.start()
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)
    cv2.imshow('Webcam', img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        new_im = Image.open('./imgs/logo.png')
        new_im.save('numToDisplay.png', 'PNG')
        displayImages=[];
        hasSelected = False
        humanDetected = False

    # TODO: ADD THIS AND FIX
    # elif cv2.waitKey(1) & 0xFF == ord('r'):
    #     displayImages=[];
    #     hasSelected = False

    # elif cv2.waitKey(1) & 0xFF == ord('2'):
    #     displayImages=[];
    #     hasSelected = False
    #     chosenRange='20'

    # elif cv2.waitKey(1) & 0xFF == ord('4'):
    #     displayImages=[];
    #     hasSelected = False
    #     chosenRange='4'

    # elif cv2.waitKey(1) & 0xFF == ord('6'):
    #     displayImages=[];
    #     hasSelected = False
    #     chosenRange='6'

    # elif cv2.waitKey(1) & 0xFF == ord('8'):
    #     displayImages=[];
    #     hasSelected = False
    #     chosenRange='8'

cap.release()
cv2.destroyAllWindows()

# https://learn.sparkfun.com/tutorials/computer-vision-and-projection-mapping-in-python/all
# CV2 PROJECT IMAGE ^
