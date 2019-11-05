import os
from bs4 import BeautifulSoup
import ftplib
import socket
import requests
import queue
from object_detection.utils.app_utils import load_yaml
import time
import random

CWD_PATH = os.getcwd()
config_folder =os.path.join(CWD_PATH, 'config.yaml')
cfg = load_yaml(config_folder)
ip = cfg['ROBOT']['IP']

sound_dir_http =cfg['ROBOT']['SOUND_DIR_HTTP']
sound_dir_ftp =cfg['ROBOT']['SOUND_DIR_FTP']

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [sound_dir_http + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def play(protocol, detected):
    if protocol == "http":
        play_sound_http(detected)
    elif protocol == "ftp":
        play_sound_ftp(detected)
    elif protocol == "local":
        play_sound_local(detected)

def play_sound(file_path):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, 7777))
    s.sendall(file_path)
    s.close()

def play_sound_local(detected):
    try:
        file_path = cfg['ROBOT']['SOUND_DIR_LOCAL']
        play_sound(file_path)
    except:
        print("smth wrong with the path")
        print(file_path)

def play_sound_http(detected):
    try:
        file_path = random.choice(listFD(sound_dir_http + detected,'mp3')).encode('utf-8')
        play_sound(file_path)
    except IndexError:
        #there is no sound for this category
        print("there is no sound for this object")

def play_sound_ftp(detected):
    ftp = ftplib.FTP(sound_dir_ftp)
    ftp.login()
    try:
        ftp.cwd(detected)
        files = ftp.nlst()
        file_name = random.choice(files)
        file_path = 'ftp://' + sound_dir_ftp + '/' + detected + '/' + file_name
        file_path = file_path.encode('utf-8')
        play_sound(file_path)
    except ftplib.error_perm:
        #there is no sound for this category
        print("there is no sound for this object")
    ftp.quit()



def sound_thread(robot_q):
    print ("Sound thread initialized")
    num_det = 0
    flag = True
    while flag:
        while not robot_q.empty():
            detected = robot_q.get()
            num_det +=1
            if (random.randrange(100) < cfg['SOUND_PROBABILITY']):
                print(num_det)
                num_det = 0
                print(detected)
                play('ftp', detected)
            if (detected == "exit"):
                print("Sound thread turned off")
                flag = False
