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
        t_start = time.time()

    if record:
        l = np.concatenate((l, q), axis=0)

    if keyboard.is_pressed('enter'):
        np.savetxt('can/test.txt', l[1:])
        record = False
        end_idx = i
        print(end_idx)
        print("len: ", end_idx - start_idx + 1)
        t_end = time.time()
        print("time: ", t_end - t_start)
        with open('can/time.txt', 'w') as f:
            f.write(str(t_end - t_start))
    elif keyboard.is_pressed('esc'):
        break
