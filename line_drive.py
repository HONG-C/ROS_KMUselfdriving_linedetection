#!/usr/bin/env python
# -*- coding: utf-8 -*-
#2021/05/29년 최종 수정됨 
import rospy
import numpy as np
import cv2, random, math, time
import matplotlib.pyplot as plt

Width = 640
Height = 480
Offset = 330
rpos=500
lpos=100
rpos_exist=1
lpos_exist=1
#low pass filter를 위한 변수 
R_sensor_value=0
R_filtered_value=0
L_sensor_value=0
L_filtered_value=0
STEER_sensor_value=0
STEER_filtered_value=0
#차선 중앙값을 확인하기 위한 변수 
center=300

#차선인식 불가 시 기본값 설정을 위한 변수 
R_turn=0
L_turn=0



#좌우 차선 및 조향각의 로우 패스 필터링을 위한 함수 
def LPF(raw_data,sensor_value,filtered_value,sensitivity=0.05):
	sensor_value=raw_data
	filtered_value=filtered_value*(1-sensitivity)+sensor_value*sensitivity
	return filtered_value

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


# 양쪽 차선을 나타내는 사각형 및 중앙점을 그리는 함수 
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 25, 15 + offset),
                       (lpos -15, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos + 5, 15 + offset),
                       (rpos + 15, 25 + offset),
                       (0, 255, 0), 2)   
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+ 5, 25 + offset),
                       (0, 0, 255), 2)   


    return img


def set_rpos(img, lines, color=[0, 0, 255], thickness=2): # rpos의 값 추출 
    if lines is not None:
	    for line in lines:
		for x1,y1,x2,y2 in line:
	 	    global rpos,rpos_prev
		    if (x1>400) and (y1==315):
			rpos=x1


		    else:
			pass
					
    return rpos


def set_lpos(img, lines, color=[0, 0, 255], thickness=2): # lpos의 값 추출 
    if lines is not None:
	    for line in lines:
		for x1,y1,x2,y2 in line:
	 	    global lpos,lpos_prev
		    if (x1<230) and (y1==315):
			lpos=x1

   		    else:
			pass

    return lpos

def line_existence(img, lines, color=[0, 0, 255], thickness=2): # 차선 감지 유무 확인 
    if lines is not None:
	    global rpos_exist
	    global lpos_exist
	    global R_turn,L_turn

	    for line in lines:
		for x1,y1,x2,y2 in line:
		    if (x1<100):
			left_rec=x1
			rpos_exist=rpos_exist+1
			if rpos_exist==500:
				print("왼쪽차선존재")
				rpos_exist=0
				R_turn=1
	
			else:
				pass
		    else:
			pass


		    if (x1>550):
			right_rec=x1
			lpos_exist=lpos_exist+1
			if lpos_exist==500:
				print("오른쪽차선존재")
				lpos_exist=0
				L_turn=1
	
			else:
				pass
   		    else:
			pass

		    if lpos_exist>=200 and rpos_exist>=200:
			print("양쪽차선존재")
			rpos_exist=0
			lpos_exist=0
			R_turn=0
			L_turn=0
		    else:
			pass




    else:
	pass

    return 


def weighted_img(img, initial_img, a=1, b=1.0, c=0.0): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)

def smoothing(lines, pre_frame):#프레임 저장 후 평균치 출력을 통한 외란 조정 
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0,0,0,0])
    
    for ii,line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

def final_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환 및 최종 차선 검출 
    global rpos,lpos,center,R_sensor_value,R_filtered_value,L_sensor_value,L_filtered_value
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)#허프변환 
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    smoothing(lines,3)#프레임 스무딩 처리 
    draw_lines(line_img, lines)

    rpos=set_rpos(line_img, lines)#rpos검출 
    lpos=set_lpos(line_img, lines)#lpos검출 
    line_existence(line_img, lines)#차선 유무 검출 
    R_filtered_value=LPF(rpos,R_sensor_value,R_filtered_value,0.8)#low pass filter를 이용해 필터링 
    L_filtered_value=LPF(lpos,L_sensor_value,L_filtered_value,0.8)#low pass filter를 이용해 필터링 
    rpos=int(R_filtered_value)
    lpos=int(L_filtered_value)   
    
    if R_turn==1:#차선 미인식 시 기본 값 설정 
	rpos=620
    if L_turn==1:
	lpos=20

    draw_rectangle(line_img, lpos, rpos,310)#차선인식 사각형 그리기 
    cv2.line(line_img,(rpos+10,325),(640,385),(0, 255, 0),4)
    cv2.line(line_img,(lpos - 20, 325),(10, 410),(0, 255, 0),4) 
    center = (lpos + rpos) / 2
    cv2.rectangle(line_img, (335, 325),
                       (345, 335),
                       (255, 0, 0), 2)   #차선 기준점 작성 

    return line_img


# You are to find "left and right position" of road lanes
def process_image(frame):
    
    return (lpos, rpos), frame

#핸들의 조작을 나타내는 함수 
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
	vertices = np.array([[(0,height),(0, height-120), (width/2-190,height/2+50),(width/2+190, height/2+50), (width,height-120) ,(width,height)]], dtype=np.int32)
	ROI_img = region_of_interest(canny_img, vertices) # ROI 설정
	hough_img = final_lines(ROI_img, 1, 1 * np.pi/180, 30, 0.01, 0.1) # 허프 변환 및 최종 라인 추출 
	result = weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap 
    	#cv2.polylines(result, [vertices], True, (255,0,0), 5)#roi 사각형 범위 출력
        steer_angle = -(center-340)/4
	STEER_filtered_value=LPF(steer_angle,STEER_sensor_value,STEER_filtered_value,0.04)#low pass filter를 이용해 필터링 
	steer_angle=STEER_filtered_value
	#조향각의 한계치를 설정하는 부분

	if steer_angle>=50:
		steer_angle=50
	if steer_angle<=-50:
		steer_angle=-50


	draw_steer(result,steer_angle)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


