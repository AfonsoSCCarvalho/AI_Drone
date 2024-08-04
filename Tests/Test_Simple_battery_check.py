from djitellopy import tello
from time import sleep
import cv2
import time

me=tello.Tello()
me.connect()
# me.streamon()
print(me.get_battery())
