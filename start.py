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


t = threading.Thread(target=obj_detection.recognition, args = (robot_q,), daemon=True)
t.start()


# In[ ]:


t2 = threading.Thread(target=mi_sounds.sound_thread, args = (robot_q,), daemon=True)
t2.start()


# In[ ]:


t3 = threading.Thread(target=mi_control.moving_thread, args = (), daemon=True)
t3.start()

