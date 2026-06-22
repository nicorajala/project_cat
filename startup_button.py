from gpiozero import Button
import subprocess
import sys
import time

button = Button(17)

print("Waiting for button press...")
button.wait_for_press()
print("Starting stream...")
subprocess.Popen([sys.executable, '/home/nico/project_cat/stream_image.py'])
