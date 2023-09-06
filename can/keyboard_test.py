import time, keyboard
import numpy as np



l = np.zeros((1, 7))
q = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, 7)
i = 0
record = False
while True:
    i += 1
    time.sleep(0.1)

    if keyboard.is_pressed('space'):
        record = True
        start_idx = i
        print(start_idx)

    if record:
        l = np.concatenate((l, q), axis=0)

    if keyboard.is_pressed('enter'):
        np.savetxt('can/test.txt', l)
        record = False
        end_idx = i
        print(end_idx)
        print(end_idx - start_idx + 1)
