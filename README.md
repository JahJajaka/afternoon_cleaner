# Afternoon cleaner
is here to talk with you from time to time.

## Overview
With this app your Xiaomi Vacuum Cleaner be able to react on the things that he encounters during his daily routine. Forget about limited voice packages. Define any sound you like to hear from your Cleaner simply adding it to the ftp folder on your computer. It will react differently on person, dog, chair or any recognizable object in your place.
In training mode you can collect annotated data for transfer learning and then train your Xiaomi Vacuum Cleaner to recognize family members or specific objects.

## Demo
coming soon...

## Requirements
1. Your Xiaomi Vacuum Cleaner V1 must be rooted. All procedures were done for V1. I'm not sure if all the same is applicable for later versions   
2. Webcam
3. Local http or ftp Server (optional)
4. Joystick for movements control (optional)


## Setup

**1. Root Xiaomi Vacuum cleaner**  

  ### Easy peasy way

  * Obtain token.  
      There are two options here. The easiest way to install MiHome_5.6.10_vevs.apk on your Android device (from robot_root folder). The apk is taken from [this guy](http://www.kapiba.ru/2017/11/mi-home.html). Once installed and linked with your Vacuum open MiHome app, go to Access -> Access to the Device. You will see token under your Vacuum name.

  * Obtain package with root access.  
      Copy v11_003532.fullos_root.pkg from robot_root folder to root folder of sdcard on your Android device. The package is taken from [here](http://4pda.ru/forum/index.php?showtopic=881982&st=5240)

  * Flash rooted firmware.  
      Install XVacuum_Firmware_3.3.0.apk (from robot_root folder) on your Android device and follow [these instructions](
    https://forum.xda-developers.com/android/apps-games/app-xvacuum-firmware-xiaomi-vacuum-t3896526). Once you have your firmware installed login to your Vacuum via ssh (user: cleaner, password: cleaner) and change the password!!! Don't make hacker's life too easy.

### Little bit more complicated way
  * Obtain token.  
      If you don't trust those evil capybara, here is another way. Install version 5.4.49 of MiHome app, then connect to your Vacuum, then find MiHome logs somewhere on your Android device and search for token.

  * Obtain package with root access.  
      If you don't trust those evil Russians follow the instructions from [here](https://github.com/dgiese/dustcloud/wiki/VacuumRobots-manual-update-root-Howto) to create the package file

  * Flash rooted firmware.  
      The [same instructions](https://github.com/dgiese/dustcloud/wiki/VacuumRobots-manual-update-root-Howto) to install created package

**2. Install requirements.**

`pip install -r requirements.txt`

IMPORTANT: python 3.6 (64bit in case of Windows), tensorflow 1.15, opencv 3.4

**3. Setup sound capabilities.**

  * Setup local http/ftp server on your local machine and share sounds directory. You may choose other files and create other folders. The idea is that each folder corresponds to detected object class.

  * On your Vacuum install sox:  

        sudo apt-get update

        sudo apt-get install sox  

        sudo apt-get install libsox-fmt-mp3

        sudo apt-get install wget


  * Grant user: cleaner rights to play sounds:

        sudo usermod -a -G audio cleaner  

        sudo reboot  

  * Copy sound_server.pl (from sounds folder) to `/usr/bin` directory of your Vacuum. Then run it:

        cd /usr/bin

        perl sound_server.pl  

**4. Edit config.yaml with your local settings.**

     VIDEO_SOURCE:  
     IP:  
     TOKEN:  
     SOUND_DIR_HTTP:  
     SOUND_DIR_FTP:  
     FAN_SPEED: #I reduce it to 1 in mi_control.py. it could be setup to 70 max  
     MODEL_NAME:  

  you may find models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.) You may use segmentation models, but they wouldn't show mask and they are too slow anyway.

**5. Open start.ipynb in Jupyter and make sure all pieces of the puzzle are in place.**  

  * Webcam is connected and recognition is working (run recognition_thread)  

  * Vacuum plays the sounds (run sound_thread), which means sound_server and http/ftp server are running. Don't forget about `SOUND_PROBABILITY` parameter from `config.yaml`. It says how often your Vacuum will be react on detected objects.  

  * Joystick is connected and you can control Vacuum with it (run moving_thread)  

  * After you satisfied with the result, kill the kernel in Jupyter.



**6. Run: `python start.py` via CLI and have fun.**


## Training mode

**1. Collecting data for training.**

  * Create label map with your own labels in `~/object_detection/data` directory. See `afternoon_cleaner_label_map.pbtxt` as example. It is important to start from id = 1.

  * Setup TRAINING_MODE parameter in `config.yaml` to 1

  * If you have subclasses, define `MAIN_CLASS` and `SUBCLASS` parameters in `config.yaml`. Default models from tensorflow model zoo do not identify subclasses. You need to collect data for only one subclass in a session i.e. if you want to train VC to recognize three persons, you need to run separate photo session for each person. If you don't have subclasses put these parameters to ''.

  * Run `object_detection` and `moving_thread`. Catch objects that you would like to train Vacuum Cleaner on. `SAVE_FRAME` parameter defines how often frame with detected object will be saved to the dataset folder. You can gain a lot of data really fast without this limitation, but it will affect variety of data.

  * All data collected in `~/object_detection/datasets/my_dataset` folder have annotations in `annotations.json` file.

  * Once you finished collecting data for all classes in your label map, run `create_ft_records.py`. It will divide your data to train and validation sets (with val_size=0.25 by default) and create tf_records from these sets.

**2. Run training.**

  * Run `python setup.py install` from `slim` folder.

  * Open `pipeline.config` in your model folder and edit `num_classes:`, `fine_tune_checkpoint:`, `label_map_path:` and `input_path:` for train and validation reader.

  * Open `train.py` and setup `'num_train_steps'` parameter. Then run `train.py`.

  * To see the progress in tensorboard. Open another command line and run `tensorboard --logdir=${MODEL_DIR}` where `${MODEL_DIR}` is the directory with train and eval datasets.
