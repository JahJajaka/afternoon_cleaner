#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import queue
import threading
import time
import mi_control
import mi_sounds
import obj_detection
robot_q = queue.Queue()


# In[ ]:


recognition_thread = threading.Thread(target=obj_detection.recognition, args = (robot_q,), daemon=True)
recognition_thread.start()


# In[ ]:


sound_thread = threading.Thread(target=mi_sounds.sound_thread, args = (robot_q,), daemon=True)
sound_thread.start()


# In[ ]:


moving_thread = threading.Thread(target=mi_control.moving_thread, args = (), daemon=True)
moving_thread.start()

