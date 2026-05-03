import time
from drift_detection import monitor

while True:
    monitor()
    time.sleep(60)