import argparse
import pandas as pd
import numpy as np
import cv2

#initialize ArgumentParser object that will parse the command line arguments into the python data types
ap = argparse.ArgumentParser()
# adding the arguments into the program
ap.add_argument("-i", "--image", required= True, help = "Image Path")
# returning the dic attributes stored in the parser and storing it is args.
args = vars(ap.parse_args())
#storing the path name that was passed in Image(key)
image_path = args["image"]

#reading the image with opencv
img = cv2.imread(image_path)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#declaring some global variables

clicked = False
r = g = b = xpos = ypos = 0
#reading the csv file and giving names to each colums
col = ["color","color_name","hex","R","G","B"]
df = pd.read_csv("/Users/MacBook/Desktop/Data_Science_Projects/Data-Science/Colour Detection/colors.csv", names=col, header = None)

#Defining the draw fucntion that will calculate the RGB values for th epixel we clicked on
#the function takes the event name( the double clicking), x,y coordinates of the mouse position. if event is double click we calculate the rgb values along with the x,y.

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global r, g, b, clicked, xpos, ypos
        clicked = True
        xpos = x
        ypos = y
        # In the case of color images, the decoded images from imread will have the channels stored in **B G R** order.
        b, g, r = img[y, x]
        # turning the RGB into integers
        b = int(b)
        g = int(g)
        r = int(r)
        
#funtion that returns the color name based on the RGB values. 
# To calculate the color we calculate the dictance d which tells us how close we are to color and choose the one having minimum distance.

#d = abs(Red – ithRedColor) + (Green – ithGreenColor) + (Blue – ithBlueColor)

def getColorName(R, G, B):
    #we chose this value randomly, sicn we will be looping throught the entire csv file to get the lowest value. we need to set it to a higher values bc we wanna end up with lowest
    minimum = 10000
    #i is the row number
    for i in range(len(df)):
        #we re looking for the minimum val bc that would be the closest to the color.
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        
        if (d<= minimum):
            #we will set munimum to that d
            minimum = d
            color_name = df.loc[i, "color name"]
    return color_name


#loop forver auntil we break the loop when we exit.   
#whenever a doube click happens, it will update the color name and the R,G,B values
#Using the cv2.imshow() function, we draw the image on the window. When the user double clicks the window, we draw a rectangle and get the color name to draw text on the window using cv2.rectangle and cv2.putText() functions.
#setting up a window to display the image names image
cv2.namedWindow('image')

#set a callback fucntion that will be called when a mouse event happens.
cv2.setMouseCallback("image", draw_function)

while(1):
    cv2.imshow("image", img)
    # cv2.waitKey(0)
    if (clicked):
        # cv2.rectangle(image, startpoint, endpoint, color, thickness) -1 thickness fills rectangle entirely
        cv2.rectangle(img, (20,20), (760,60), (r,g,b), -1
                      )
        
        #Creating text string to display ( Color name and RGB values )
        text = getColorName(r, g, b) + "R="+ str(r) + "G="+ str(g)+ "B="+ str(b)
        
        #cv2.putText(img,text,start,font(0-7), fontScale, color, thickness, lineType, (optional bottomLeft bool) )
        cv2.putText(img, text, (50, 50), 2, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        #For very light colours we will display text in black colour
        if(r+g+b >= 600):
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
         
        clicked = False
    #It waits for 20 milliseconds for key presses and checks whether the pressed key is esc.
    if cv2.waitKey(20) & 0xFF == 27:
        break  
        
cv2.destroyAllWindows()