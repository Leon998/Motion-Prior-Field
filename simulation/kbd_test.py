import keyboard

while True:
    print("running")
    if keyboard.is_pressed('w'):
        print('Forward')
    elif keyboard.is_pressed('s'):
        print('Backward')
    elif keyboard.is_pressed('a'):
        print('Left')
    elif keyboard.is_pressed('d'):
        print('Right')
    elif keyboard.is_pressed('enter'):  # if key 'enter' is pressed 
        print('You pressed enter!')
    elif keyboard.is_pressed('q'):
        print('Quit!')
        break
