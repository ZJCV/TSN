# -*- coding: utf-8 -*-

"""
@date: 2020/11/9 上午11:07
@file: live.py
@author: zj
@description: 
"""

import cv2
import time
from multiprocessing import Process, Queue
import subprocess as sp


class Live(object):
    frame = None

    def __init__(self, enable, way, url, size=(1280, 720), fps=25):
        self.enable = enable
        if not enable:
            return
        self.frame_queue = Queue(maxsize=5)
        self.fps = fps
        self.size = size

        if way == "rtmp":
            self.command = ['ffmpeg',
                            '-re',
                            '-loglevel', 'error',
                            '-y',
                            '-f', 'rawvideo',
                            '-vcodec', 'rawvideo',
                            '-pix_fmt', 'bgr24',
                            '-s', "{}x{}".format(*self.size),
                            '-r', str(fps),
                            '-i', '-',
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-preset', 'ultrafast',
                            # '-preset', 'veryfast',
                            '-f', 'flv',
                            url]
        elif way == "rtsp":
            self.command = ['ffmpeg',
                            '-loglevel', 'error',
                            '-y',
                            '-f', 'rawvideo',
                            '-vcodec', 'rawvideo',
                            '-pix_fmt', 'bgr24',
                            '-s', "{}x{}".format(*self.size),
                            '-r', str(fps),
                            '-i', '-',
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-preset', 'ultrafast',
                            '-rtsp_transport', 'tcp',
                            '-f', 'rtsp',
                            url]

    def read_frame(self, view):
        if not self.enable:
            return
            # print("开启推流")
        frame = cv2.resize(view, self.size)
        # put frame into queue
        if self.frame_queue.full():
            self.frame_queue.get()
            self.frame_queue.put(frame)
        else:
            self.frame_queue.put(frame)

        # time.sleep(1 / 12)
        # self.p.stdin.write(frame.tostring())

    def push_frame(self, queue):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break

        now_frame = None
        while True:
            if queue.empty() is not True:
                frame = queue.get()
                now_frame = frame
                # write to pipe
                try:
                    p.stdin.write(frame.tostring())
                except:
                    pass
                time.sleep(1 / self.fps)
            elif now_frame is not None:
                try:
                    p.stdin.write(now_frame.tostring())
                except:
                    pass
                time.sleep(1 / self.fps)

    def run(self):
        # threads = [
        #     # threading.Thread(target=self.read_frame),
        #     threading.Thread(target=self.push_frame),
        # ]
        # [thread.setDaemon(True) for thread in threads]
        # [thread.start() for thread in threads]

        # self.process = threading.Thread(target=self.push_frame, args=(self.frame_queue,))
        # self.process.setDaemon(True)
        # self.process.start()

        if self.enable:
            # self.process = threading.Thread(target=self.push_frame, args=(self.frame_queue,))
            # self.process.setDaemon(True)
            # self.process.start()
            self.process = Process(target=self.push_frame, args=(self.frame_queue,))
            self.process.daemon = True
            self.process.start()
