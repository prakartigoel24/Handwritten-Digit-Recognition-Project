import numpy as np
from numpy import *
from tokenize import Number
import cv2
from numpy.lib.type_check import imag
from keras.models import load_model
from pygame import image
import pygame, sys
from pygame.locals import *
from tensorflow.python.keras.backend import constant

WindowsizeX = 650
WindowsizeY = 500
BoundaryInc = 5

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0)

IMAGESAVE= False
Model = load_model("digits.model")
LABELS = {
    0:"Zero" , 1:"One" , 2:"Two", 3:"Three" , 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"
}
pygame.init()

DISPLAYSURF = pygame.display.set_mode((WindowsizeX, WindowsizeY))
FONT = pygame.font.Font("FreeSansBold.ttf", 18)

pygame.display.set_caption("DIGIT SCREEN")

iswriting  = False
number_xcord=[]
number_ycord=[]
image_cnt =1
PREDICT = True
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord , ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting =False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x , rect_max_x = max(number_xcord[0]-BoundaryInc, 0),min(WindowsizeX, number_xcord[-1]+BoundaryInc)
            rect_min_Y , rect_max_Y = max(number_ycord[0]-BoundaryInc,0),min(number_ycord[-1]+BoundaryInc,WindowsizeX)
            
            number_xcord=[]
            number_ycord=[]

            img_array = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt +=1

            if PREDICT:
                image = cv2.resize(img_array, (28,28))
                image = np.pad(image, (10, 10),'constant', constant_values = 0)
                image = cv2.resize(image, (28,28))/255
                img  =image.reshape(1, 28, 28, 1)
                label = str(LABELS[np.argmax(Model.predict(img))])
                textsurface = FONT.render(label, True,RED, WHITE)
                textRecObj = textsurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y

                DISPLAYSURF.blit(textsurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == 'c':
                DISPLAYSURF.fill(BLACK)
        pygame.display.update()           