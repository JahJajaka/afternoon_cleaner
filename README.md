# Afternoon cleaner
is here to talk with you from time to time.

## Overview
With this app your Xiaomi Vacuum Cleaner be able to react on the things that he encounters during his daily routine. Forget about limited voice packages. Define any sound you like to hear from your Cleaner simply adding it to the ftp folder on your computer. It will react differently on person, dog, chair or any recognizable object in your place. 

## Demo
coming soon...

## Requirements
1. Your Xiaomi Vacuum Cleaner V1 must be rooted. All procedures were done for V1. I'm not sure if all the same is applicable for later versions   
2. Webcam
3. Local http or ftp Server (optional)
4. Joystick for movements control (optional)


## Setup
1. Root Xiaomi Vacuum cleaner
### Easy peasy way
  a. Obtain token.
    There are two options here. The easiest way to install MiHome_5.6.10_vevs.apk on your Android device (from robot_root folder). The apk is taken from this guy (http://www.kapiba.ru/2017/11/mi-home.html). Once installed and linked with your Vacuum open MiHome app, go to Access -> Access to the Device. You will see token under your Vacuum name.
  b. Obtain package with root access.
    Copy v11_003532.fullos_root.pkg from robot_root folder to root folder of sdcard on your Android device. The package is taken from here (http://4pda.ru/forum/index.php?showtopic=881982&st=5240)
  c. Flash rooted firmware
    Install XVacuum_Firmware_3.3.0.apk (from robot_root folder) on your Android device and follow these instructions:
    https://forum.xda-developers.com/android/apps-games/app-xvacuum-firmware-xiaomi-vacuum-t3896526
    once you have your firmware installed login to your Vacuum via ssh (user: cleaner, password: cleaner) and change the password!!! Don't make hacker's life too easy.
### Little bit more complicated way
  a. Obtain token.
    If you don't trust those evil capybara, here is another way. Install version 5.4.49 of MiHome app, then connect to your Vacuum, then find MiHome logs somewhere on your Android device and search for token.    
  b. Obtain package with root access.
    If you don't trust those evil Russians follow the instructions from here: https://github.com/dgiese/dustcloud/wiki/VacuumRobots-manual-update-root-Howto to create the package file
  c. Flash rooted firmware
    The same instructions:  https://github.com/dgiese/dustcloud/wiki/VacuumRobots-manual-update-root-Howto to install created package

2. Install requirements.
  pip install requirements.txt
  IMPORTANT: tensorflow 1.15, opencv 3.4

3. Setup sound capabilities
  a. Setup local http/ftp server on your local machine and share sounds directory. You may choose other files and create other folders. The idea is that each folder corresponds to detected object class.
  b. On your Vacuum install sox:
      sudo apt-get update
      sudo apt-get install sox
      sudo apt-get install libsox-fmt-mp3
      sudo apt-get install wget
  c. Grant user: cleaner rights to play sounds:
      sudo usermod -a -G audio cleaner
      sudo reboot
  d. Copy sound_server.pl (from sounds folder) to /usr/bin directory of your Vacuum. Then run it:
      cd /usr/bin
      perl sound_server.pl

4. Edit config.yaml with your local settings:
   VIDEO_SOURCE:
   IP:
   TOKEN:
   SOUND_DIR_HTTP:
   SOUND_DIR_FTP:
   FAN_SPEED: #I reduce it to 1 in mi_control.py. it could be setup to 70 max
   MODEL_NAME:

    you may find models here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md. You may use segmentation models, but they wouldn't show mask and they are too slow anyway.

5. Open start.ipynb in Jupyter and make sure all pieces of the puzzle are in place
  a. Webcam is connected and recognition is working (run recognition_thread)
  b. Vacuum plays the sounds (run sound_thread), which means sound_server and http/ftp server are running. Don't forget about SOUND_PROBABILITY parameter from config.yaml. It says how often your Vacuum will be react on detected objects.
  c. Joystick is connected and you can control Vacuum with it (run moving_thread).

6. Run start.py and have fun.
