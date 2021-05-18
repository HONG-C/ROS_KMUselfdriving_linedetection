#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time
import matplotlib.pyplot as plt

Width = 640#640
Height = 480
Offset = 330
rpos=500
rpos_prev=500
lpos=100
lpos_prev=100
center=300
x_axis=[]
y_axis=[]
running_time=0

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    if lines is not None:
	    for line in lines:
		for x1,y1,x2,y2 in line:
		    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)   
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+ 5, 25 + offset),
                       (0, 0, 255), 2)   


    return img


def draw_moving_rectangle_R(img, lines, color=[0, 0, 255], thickness=2): # 기준점 그리기
    if lines is not None:
	    for line in lines:
		for x1,y1,x2,y2 in line:
	 	    global rpos,rpos_prev
		    if (x1>400) and (y1==315):
			rpos=x1
			#if abs(rpos-rpos_prev)<5:
				#rpos=rpos_prev
    			cv2.rectangle(img, (rpos +10, 325),
                       (rpos + 20, 335),
                       (0, 255, 0), 2)
   			#cv2.line(img,(rpos+10,345),(rpos+10,345),(0, 255, 0),10)

		    else:
			if abs(rpos-rpos_prev)<5:
				rpos=rpos_prev
					
    return rpos


def draw_moving_rectangle_L(img, lines, color=[0, 0, 255], thickness=2): # 기준점 그리기
    if lines is not None:
	    for line in lines:
		for x1,y1,x2,y2 in line:
	 	    global lpos,lpos_prev
		    if (x1<200 and (y1==315)):
			lpos=x1
			#if abs(lpos-lpos_prev)<5:
				#lpos=lpos_prev
    			cv2.rectangle(img, (lpos - 20, 325),
                       (lpos -10, 335),
                       (0, 255, 0), 2)
   		    else:
			if abs(lpos-lpos_prev)<5:
				lpos=lpos_prev


    return lpos


def weighted_img(img, initial_img, a=1, b=1.0, c=0.0): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)

def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0,0,0,0])
    
    for ii,line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    global rpos,lpos,center
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    rpos=draw_moving_rectangle_R(line_img, lines)
    lpos=draw_moving_rectangle_L(line_img, lines)
    smoothing(lines,10)
    center = (lpos + rpos) / 2
    cv2.rectangle(line_img, (center-5, 345),
                       (center+ 5, 355),
                       (0, 0, 255), 2)   #rpos,lpos기준 중심점 작성 
    cv2.rectangle(line_img, (335, 345),
                       (345, 355),
                       (255, 0, 0), 2)   #차선 기준점 작성 
    print("right:",rpos,"left:",lpos)	 

    return line_img





# You are to find "left and right position" of road lanes
def process_image(frame):
    global Offset
   
    #frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    
    return (lpos, rpos), frame


def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728

    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 1.5, 0.7)    
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    cv2.imshow('steer', image)

# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)

    while not rospy.is_shutdown():
        ret, image = cap.read()
        pos, frame = process_image(image)
	height, width = image.shape[:2] # 이미지 높이, 너비
	gray_img = grayscale(image) # 흑백이미지로 변환
	blur_img = gaussian_blur(gray_img, 3) # Blur 효과
	canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘
	vertices = np.array([[(0,height),(0,height-120),(width/2-200, height/2+50), (width/2+200, height/2+50), (width,height-120) ,(width,height)]], dtype=np.int32)
	#vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
	ROI_img = region_of_interest(canny_img, vertices) # ROI 설정
	hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 0.001, 0.01) # 허프 변환
	result = weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap
	cv2.imshow('hough_img',hough_img) # roi 이미지 출력     
    
	cv2.polylines(result, [vertices], True, (255,0,0), 5)#roi 사각형 범위 출력

 
        steer_angle = -(center-340)/2
        #print("steer_angle:",steer_angle)
        draw_steer(result, steer_angle)
	rpos_prev=rpos
	lpos_prev=lpos
#차선 인지 확인을 위한 그래프 작성 
	running_time=running_time+1
	x_axis.append(running_time)
	y_axis.append(rpos)
	plt.plot(x_axis,y_axis)
	#plt.pause(0.0000001)
        #바로 위에 꺼 주석처리 해제하면 그래프 확인가능 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


