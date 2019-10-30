#!/usr/bin/env python
# coding: utf-8

# In[8]:

import os
import random
import miio
import pygame
import time
import queue
import threading
import time

from threading import Thread
from object_detection.utils.app_utils import load_yaml

CWD_PATH = os.getcwd()
config_folder =os.path.join(CWD_PATH, 'config.yaml')
cfg = load_yaml(config_folder)
ip = cfg['ROBOT']['IP']
token = cfg['ROBOT']['TOKEN']

BUTTON_SQUARE = 0
BUTTON_X = 1
BUTTON_CIRCLE = 2
BUTTON_TRIANGLE = 3




def init_joystick():
    pygame.init()
    pygame.joystick.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()
    return controller

def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def move_robot(bot, buttons, axis):
    rot = 0
    val = 0
    to_min, to_max = -0.3, 0.3
    # Right stick X
    if axis[2] != 0:
        rot = -translate(axis[2], -1, 1, -90, 90)
        if abs(rot) < 8:
            rot = 0
    # Left stick Y, -1 up, 1 down
    if axis[1] != 0:
        val = -translate(axis[1], -1, 1, to_min, to_max)
        if abs(val) < 0.07:
            val = 0
    if rot or val:
        bot.manual_control(rot, val, 150)


def moving_thread():
    controller = init_joystick()
    bot = miio.vacuum.Vacuum(ip, token)
    bot.set_fan_speed(1)
    modes = ['manual', 'home', 'spot', 'cleaning', 'unk']
    mode = 'unk'
    axis = [0.00 for _ in range(6)]
    flag = True
    button = [False for _ in range(14)]
    print('Press start to start!')
    while flag:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                button[event.button] = True
                # Touchpad to exit
                if event.button == 13:
                    bot.set_fan_speed(30)
                    flag = False
            elif event.type == pygame.JOYBUTTONUP:
                if mode == 'unk':
                    print('Ready to go! Press X to start manual mode')
                    if event.button == BUTTON_X:
                        mode = 'manual'
                        bot.manual_start()
                elif mode == 'manual':
                    if event.button == BUTTON_TRIANGLE:
                        bot.manual_stop()
                        mode = 'unk'
                    elif event.button == BUTTON_X:
                        pass
                    elif event.button == BUTTON_CIRCLE:
                        pass

        if mode == 'manual':
            try:
                move_robot(bot, button, axis)  # see ya in the next step
            except:
                bot.manual_start()
                pass
        time.sleep(0.01)
