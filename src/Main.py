# coding=utf-8
import os
import json, time
import threading
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QSizePolicy
from qt_thread_updater import get_updater
from src import config as co

def text_size(frame):
    frame_height, frame_width = frame.shape[:2]

    # Define the text and initial font settings
    text = "Computer"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate font scale and thickness based on frame size
    font_scale = frame_width / 600  # Adjust this divisor to scale text size (smaller values = larger text)
    font_thickness = max(2, int(frame_height / 200))  # Adjust divisor for thickness (ensure thickness >= 1)
    text_color = (0, 165, 255)  # Orange in BGR

    # Calculate text size with the dynamic scale and thickness
    (_, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Define text position in the top-left corner with padding
    padding = int(0.02 * frame_height)  # 2% of the frame height as padding
    text_x = padding
    text_y = text_height + padding  # Position Y to account for text height

    return (text_x, text_y), font, font_scale, text_color, font_thickness

class Main:
    def __init__(self, MainGUI):
        self.MainGUI = MainGUI
        self.camera = None
        self.image = None
        self.ret = False
        self.start_camera = True
        self.init_text_size()

    def img_cv_2_qt(self, img_cv):
        height, width, channel = img_cv.shape
        bytes_per_line = channel * width
        img_qt = QtGui.QImage(img_cv, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        return QtGui.QPixmap.fromImage(img_qt)
    
    def init_devices(self, url_camera):
        # Create empty buffer
        self.frame_buffer = np.empty((0, *self.dim, self.channels))
        self.camera = cv2.VideoCapture(url_camera)
        self.camera.set(cv2.CAP_PROP_FPS, 25)  # set the FPS to 25 
        self.ret, frame = self.camera.read()
        if not self.ret:
            self.start_camera = False
            self.MainGUI.MessageBox_signal.emit("Có lỗi xảy ra ! \n Không tìm thấy camera/video", "error")
        else:
            self.start_camera = True
            (self.text_x, self.text_y), self.font, self.font_scale, self.text_color, self.font_thickness = text_size(frame)

    def auto_camera(self):
        url_camera = co.CAMERA_DEVICE
        self.init_devices(url_camera)
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    self.image = frame.copy()
                    # pre-process each frame
                    frame_res = cv2.resize(self.image, self.dim)
                    frame_res = frame_res / 255.0
                else:
                    get_updater().call_latest(self.MainGUI.text_result.setText, "")
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()

    def auto_video(self, path_video):
        url_camera = path_video
        self.init_devices(url_camera)
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    self.image = frame.copy()
                    # pre-process each frame
                    frame_res = cv2.resize(self.image, self.dim)
                    frame_res = frame_res / 255.0
                else:
                    get_updater().call_latest(self.MainGUI.text_result.setText, "")
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()
      
    def capture_image(self):
        if self.image is not None and self.ret and self.start_camera:
            image = self.image.copy()
            # Call SLVR model
            self.close_camera()
            get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
        else:
            self.MainGUI.MessageBox_signal.emit("Không tìm thấy Camera/Video !", "error")
    
    def manual_image(self, image_file):
        image = cv2.imread(image_file)
        # Call SLVR model

        get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))


    def close_camera(self):
        try:
            self.start_camera = False
            if self.ret:
                self.camera.release()
            self.camera = None
            self.ret = False
            
            time.sleep(1)
            self.MainGUI.label_Image.clear()

        except Exception as e:
                print("Bug: ", e)

    def init_text_size(self):
        self.text_x = 20
        self.text_y = 20
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.text_color = (0, 255, 0)
        self.font_thickness = 2