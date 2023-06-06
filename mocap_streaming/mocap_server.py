import sys
import time
from NatNetClient import NatNetClient
import redis
import pickle
from functools import partial
import numpy as np




# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame(r, new_id, position, rotation):
    # print("Received frame for rigid body", new_id)
    # print("Position:", position)
    # print("Rotation", rotation)
    if new_id == 1:
        r.set('hand_position', str(position), ex=1)
        r.set('hand_rotation', str(rotation), ex=1)
    else:
        r.set('object_position', str(position), ex=1)
        r.set('object_rotation', str(rotation), ex=1)


if __name__ == "__main__":

    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = True

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    rds = redis.Redis(connection_pool=pool)

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.rigid_body_listener = partial(receive_rigid_body_frame, rds)

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    is_looping = True
    while is_looping:
        inchars = input('Enter command or (\'h\' for list of commands)\n')
        if len(inchars) > 0:
            c1 = inchars[0].lower()
            if c1 == 'q':
                is_looping = False
                streaming_client.shutdown()
                break
            else:
                print("Error: Command %s not recognized" % c1)
            print("Ready...\n")
    print("exiting")