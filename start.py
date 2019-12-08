
import queue
import threading
import time
import mi_control
import mi_sounds
import obj_detection
import concurrent.futures




if __name__ == "__main__":
    robot_q = queue.Queue()
    photo_q = queue.Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.submit(obj_detection.recognition, robot_q, photo_q)
        executor.submit(mi_sounds.sound_thread, robot_q)
        executor.submit(mi_control.moving_thread, photo_q)
