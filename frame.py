import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

def canny_edge(frame): 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray_frame, 50,150)
    return canny

def segment(frame): 
    height = frame.shape[0]
    polygon = np.array([
        [(0,height), (800,height), (400,250)]
    ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygon, 255)
    croped = cv2.bitwise_and(frame, mask)
    return croped

def calculate_lines(frame, hough_lines): 
    left, right  = [], [] 
    for line in hough_lines: 
        x1,y1,x2,y2 = line.reshape(4)
        par = np.polyfit((x1,x2),(y1,y2),1)
        slope = par[0]
        intercept = par[1]
        if slope < 0: 
            left.append((slope,intercept))
        else:
            right.append((slope,intercept))
    left_avg = np.average(left,axis= 0)
    right_avg = np.average(right,axis= 0)
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, par): 
    slope, inter = par
    y1 = frame.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - inter)/ slope)
    x2 = int((y2 - inter)/ slope)
    return np.array([x1, y1, x2, y2])

def visualize(frame, lines): 
    lines_viz = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lines_viz, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_viz

vidcap = cv2.VideoCapture('input.mp4')
while (vidcap.isOpened()):
    ret, frame = vidcap.read()
    x = canny_edge(frame)
    # cv2.imshow("canny",x)
    croped = segment(x)
    hough_lines = cv2.HoughLinesP(croped, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    # print(hough_lines)
    lines = calculate_lines(frame, hough_lines)
    lines_visualize = visualize(frame, lines)
    # cv2.imshow("hough", lines_visualize)
    output = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    cv2.imshow("image",output)
    # print(x.shape[0])
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vidcap.release()
cv2.destroyAllWindows()



