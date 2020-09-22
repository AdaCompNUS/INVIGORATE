import numpy as np
import cv2

def Get_Image_Co(Img, Dep):
    Img_Co = []
    Img_C = Img.copy()
    State = 0
    x_l = 0
    y_l = 0
    x_r = 0
    y_r = 0
    List = {'Img_Co':Img_Co,
            'Img_C':Img_C,
            'State':State,
            'x_l':x_l,
            'y_l':y_l,
            'x_r':x_r,
            'y_r':y_r,
            'Img':Img,
            'Dep':Dep.astype(np.float32)}
    def get_One_Co(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            List['State'] = 1
            List['x_l'] = x
            List['y_l'] = y
        if event == cv2.EVENT_MOUSEMOVE and List['State'] == 1:
            List['Img_C'] = List['Img'].copy()
            cv2.rectangle(List['Img_C'],(List['x_l'], List['y_l']),(x, y),(55,255,155),1)
        if event == cv2.EVENT_LBUTTONUP:
            List['State'] = 0
            List['x_r'] = x
            List['y_r'] = y
            cv2.rectangle(List['Img'],(List['x_l'], List['y_l']),(List['x_r'], List['y_r']),(55,255,155),1)
            List['Img_C'] = List['Img'].copy()
            s = 0
            S = 0
            for i in range(List['y_l'], List['y_r']):
                for j in range(List['x_l'], List['x_r']):
                    if Dep[i][j] != 0.:
                        S += Dep[i][j]
                        s += 1
            if s == 0:
             print(List['x_l'], List['x_r'], List['y_l'], List['y_r'])
            print(List['x_l'], List['x_r'], List['y_l'], List['y_r'])
            S = S / s

            List['Img_Co'].append([(float(List['x_l']) + float(List['x_r'])) / 2,
                                   (float(List['y_l']) + float(List['y_r'])) / 2,
                                   S])
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',get_One_Co)

    while(1):
        cv2.imshow('image',List['Img_C'])
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    Img_Co = np.array(List['Img_Co'])
    return Img_Co

