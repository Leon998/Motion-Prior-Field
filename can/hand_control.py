from ctypes import *
import time, keyboard
import numpy as np
 
VCI_USBCAN2 = 4
STATUS_OK = 1
class VCI_INIT_CONFIG(Structure):  
    _fields_ = [("AccCode", c_uint),
                ("AccMask", c_uint),
                ("Reserved", c_uint),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)
                ]  
class VCI_CAN_OBJ(Structure):  
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte*8),
                ("Reserved", c_ubyte*3)
                ] 
 
CanDLLName = 'can/ControlCAN.dll' #把DLL放到对应的目录下
canDLL = windll.LoadLibrary(CanDLLName)

 
ret = canDLL.VCI_OpenDevice(VCI_USBCAN2, 0, 0)
 
#初始0通道
vci_initconfig = VCI_INIT_CONFIG(0x80000008, 0xFFFFFFFF, 0,
                                 0, 0x00, 0x1C, 0)#波特率500k，正常模式
ret = canDLL.VCI_InitCAN(VCI_USBCAN2, 0, 0, byref(vci_initconfig))
 
ret = canDLL.VCI_StartCAN(VCI_USBCAN2, 0, 0)
 
#初始1通道
ret = canDLL.VCI_InitCAN(VCI_USBCAN2, 0, 1, byref(vci_initconfig))
 
ret = canDLL.VCI_StartCAN(VCI_USBCAN2, 0, 1)

ubyte_array = c_ubyte*8
ubyte_3array = c_ubyte*3

# 接收数据
# 结构体数组类
import ctypes
class VCI_CAN_OBJ_ARRAY(Structure):
    _fields_ = [('SIZE', ctypes.c_uint16), ('STRUCT_ARRAY', ctypes.POINTER(VCI_CAN_OBJ))]
    def __init__(self,num_of_structs):
                                                             #这个括号不能少
        self.STRUCT_ARRAY = ctypes.cast((VCI_CAN_OBJ * num_of_structs)(),ctypes.POINTER(VCI_CAN_OBJ))#结构体数组
        self.SIZE = num_of_structs#结构体长度
        self.ADDR = self.STRUCT_ARRAY[0]#结构体数组地址  byref()转c地址

rx_vci_can_obj = VCI_CAN_OBJ_ARRAY(2500)#结构体数组


def wrist_limit(flexion_degree, rotation_degree):
    if flexion_degree < -30:
        flexion_degree = -30
    elif flexion_degree > 30:
        flexion_degree = 30
    if rotation_degree < -45:
        rotation_degree = -45
    elif rotation_degree > 45:
        rotation_degree = 45

    return int(flexion_degree), int(rotation_degree)
    

def wrist_tf(flexion_degree=0, rotation_degree=0):
    wrist_rotation = rotation_degree if rotation_degree >= 0 else -rotation_degree + 128
    wrist_flexion = flexion_degree if flexion_degree >= 0 else -flexion_degree + 128
    a = ubyte_array(2, 0, wrist_flexion, wrist_rotation)
    b = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141314, 0, 0, 1, 0, 1, 4, a, b)
 
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)

def read_wrist():
    ret = canDLL.VCI_Receive(VCI_USBCAN2, 0, 0, byref(rx_vci_can_obj.ADDR), 2500, 0) 
    while True:#如果没有接收到数据，一直循环查询接收。
        ret = canDLL.VCI_Receive(VCI_USBCAN2, 0, 0, byref(rx_vci_can_obj.ADDR), 2500, 0)
        if (ret > 0) & (rx_vci_can_obj.ADDR.ID == 0x12345700):
            joint_data = list(rx_vci_can_obj.ADDR.Data)
            wrist_flexion = joint_data[0]
            wrist_rotation = joint_data[1]
            rotation_degree = wrist_rotation if wrist_rotation <= 128 else -wrist_rotation + 128
            flexion_degree = wrist_flexion if wrist_flexion <= 128 else -wrist_flexion + 128
            # print(flexion_degree, rotation_degree)
            break
    return flexion_degree, rotation_degree


def hand_tf(name, action):
    config_list = [0xAA, 0x55, 0x4D, name, action]
    check_bit = sum(config_list) & 0xff
    c = (0xAA, 0x55, 0x4D, name, action, check_bit)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 6, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1) 

def grasp_handle():
    """
    Grasp type only for mug handle
    """
    c = (0xAA, 0x55, 0x06, 0x66, 0x03, 0xff, 0x03, 0x88)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    time.sleep(1.5)
    c = (0xAA, 0x55, 0x04, 0x88, 0x03, 0xff, 0x01, 0x88)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)

def grasp_thin():
    c = (0xAA, 0x55, 0x04, 0x44, 0x05, 0x00, 0x02, 0x00)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    # time.sleep(1.5)

def grasp_medium():
    c = (0xAA, 0x55, 0x05, 0x44, 0x05, 0x00, 0x02, 0x00)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    # time.sleep(1.5)

def grasp_other():
    """
    General grasp shape
    """
    c = (0xAA, 0x55, 0x06, 0x00, 0x05, 0x88, 0x03, 0x00)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    # time.sleep(1.5)

def grasp_pitcher():
    c = (0xAA, 0x55, 0x02, 0x00, 0x02, 0x00, 0x02, 0xDD)
    d = ubyte_3array(0, 0, 0)
    vci_can_obj = VCI_CAN_OBJ(0x13141316, 0, 0, 1, 0, 1, 8, c, d)
    ret = canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    # time.sleep(1.5)

def release_grasp(t=0):
    if t != 0:
        time.sleep(t)
    hand_tf(0xA1, 0x01)

if __name__ == "__main__":
    flexion_degree, rotation_degree = 0, 0
    while True:
        if keyboard.is_pressed('ctrl'):
            wrist_tf(0, -45)
            time.sleep(2.5)
            wrist_tf(0, 45)
            time.sleep(2.5)
            flexion_degree, rotation_degree = read_wrist()
        elif keyboard.is_pressed('backspace'):
            wrist_tf(30, 45)
            time.sleep(1.5)
            flexion_degree, rotation_degree = read_wrist()
        elif keyboard.is_pressed('1'):
            grasp_handle()
        elif keyboard.is_pressed('2'):
            grasp_thin()
        elif keyboard.is_pressed('3'):
            grasp_medium()
        elif keyboard.is_pressed('4'):
            grasp_other()
        elif keyboard.is_pressed('5'):
            grasp_pitcher()
        elif keyboard.is_pressed('space'):
            # release_grasp()
            hand_tf(0xA1, 0x01)
        elif keyboard.is_pressed('esc'):
            break
        
        print(flexion_degree, rotation_degree)


    canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 

    

