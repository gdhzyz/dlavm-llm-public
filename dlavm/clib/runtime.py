import os
import ctypes
import numpy as np

def init():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rt_lib = ctypes.CDLL(os.path.join(current_dir, ".", "PCIeDriver.dll"), ctypes.RTLD_GLOBAL)

    _open_device, _CSB_Write, _CSB_Read = rt_lib.open_device, rt_lib.CSB_Write, rt_lib.CSB_Read
    _init, _DDR_Read, _DDR_Write = rt_lib.init, rt_lib.DDR_Read, rt_lib.DDR_Write
    _open_device.restype = ctypes.POINTER(ctypes.c_uint64)
    _DDR_Read.restype = ctypes.POINTER(ctypes.c_uint8)

    devices = _open_device()
    accel, c2hx, h2cx = devices[0], devices[2], devices[6]

    def CSB_Write(address, data):
        _CSB_Write(accel, address, data)

    def CSB_Read(address):
        return _CSB_Read(accel, address)

    def DDR_Write(np_data, address):
        bdata = np_data.tobytes()
        data_size = len(bdata)
        brpt = (ctypes.c_byte * data_size).from_buffer(bytearray(bdata))
        trpt = ctypes.cast(brpt, ctypes.POINTER(ctypes.c_uint8))
        tp_r0 = _DDR_Write(h2cx, trpt, ctypes.c_uint64(address), data_size)
        if tp_r0 == data_size:
            return 1
        else:
            return 0
        
    def DDR_Read(address, shape, dtype):
        data_size = len(np.zeros(shape=shape, dtype=dtype).tobytes())
        res = bytearray(data_size)
        uint8_handle = _DDR_Read(c2hx, ctypes.c_uint64(address), data_size)
        rptr = (ctypes.c_byte * data_size).from_buffer(res)
        if not ctypes.memmove(rptr, uint8_handle, data_size):
            raise RuntimeError("memmove faild")
        data = np.frombuffer(res, dtype=dtype).reshape(shape)
        return data

    return CSB_Read, CSB_Write, DDR_Read, DDR_Write