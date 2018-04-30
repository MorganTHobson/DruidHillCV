import tkinter as tk, threading
import os, sys, time
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
import imageio
import pedestrian_gui

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import csv

video_name = "../video_input/IMG_1295.MOV"
title = "Druid Hill Project"

# define time format and output header -> TODO: add GPS LOCATION
output = ["Time", "Type", "Direction", "Total"]
SKIP_FRAMES = 10

class DruidGUI(object):

    def __init__(self, master, cap, title):
        self.master = master
        self.cap = cap

        master.title(title)
        master.resizable(width=tk.TRUE, height=tk.TRUE)
        master.geometry('1024x768')
        master.config(background="black")

        # top frame for labels and description
        self.topframe = tk.Frame(master, width=1024, height=50, bg="brown")
        self.topframe.pack()
        self.topframe.pack_propagate(0)
        
        self.w = tk.Label(self.topframe, text="Druid Hill Park Detector", bg="brown", fg="white")
        self.w.pack()

        self.videoarea = tk.Frame(master, width=1024, height=668)
        self.videoarea.pack()
        self.videoarea.pack_propagate(0)

        self.vframe = tk.Label(self.videoarea)
        self.vframe.pack()
        self.vframe.pack_propagate(0)

        # bottom frame for buttons
        self.bottomframe = tk.Frame(master, width=1024, height=50, bg="brown")
        self.bottomframe.pack()
        self.bottomframe.pack_propagate(0)

        self.startbutton = tk.Button(self.bottomframe, text="start", command=self.startthread)
        self.startbutton.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.stopbutton = tk.Button(self.bottomframe, text="stop", command=self.stopthread)
        self.stopbutton.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.quitbutton = tk.Button(self.bottomframe, text="quit", command=master.destroy)
        self.quitbutton.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.detectorbutton = tk.Button(self.bottomframe, text="show feed analysis", bg = "green", command=pedestrian_gui.analyze_vid)
        self.detectorbutton.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.closedetectorbutton = tk.Button(self.bottomframe, text="close feed analysis", bg = "green", command=self.pedestrian_gui_destroy)
        self.closedetectorbutton.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

    def startthread(self):
        self.thread = threading.Thread(target=self.startfeed)
        self.thread.daemon = 1
        self.thread.start()

    def stopthread(self):
        print("stop: not implemented yet")
        pass

    def startfeed(self):
        _, myframe = self.cap.read()
        cv2image = cv2.cvtColor(myframe, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.vframe.imgtk = imgtk
        self.vframe.configure(image=imgtk)
        self.vframe.after(20, self.startfeed)

    def stopfeed(self, event):
        print(event)

    def pedestrian_gui_analyze_vid(self):
        # print("gui analyze: not implemented yet")
        # pass
        pedestrian_gui.analyze_vid(video_name)

    def pedestrian_gui_destroy(self):
        print("gui destroy: not implemented yet")
        pass


# start GUI
master = tk.Tk()
cap = cv2.VideoCapture(video_name)
druid_gui = DruidGUI(master, cap, title)

master.mainloop()
