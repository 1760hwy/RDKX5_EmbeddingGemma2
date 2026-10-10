"""Numpy adapter to the board-built persistent DNN helper."""
import ctypes as C
from pathlib import Path
import numpy as np

class Runtime:
    def __init__(self, model, shapes):
        lib=Path(__file__).resolve().parents[1]/'bpu/runtime/libembedding_bpu.so'
        self.lib=C.CDLL(str(lib))
        self.lib.bpu_last_error.restype=C.c_char_p
        self.lib.bpu_create.argtypes=[C.c_char_p];self.lib.bpu_create.restype=C.c_void_p
        self.lib.bpu_close.argtypes=[C.c_void_p]
        self.lib.bpu_shape.argtypes=[C.c_void_p,C.c_int,C.c_int,C.POINTER(C.c_int)]
        self.lib.bpu_run.argtypes=[C.c_void_p,C.POINTER(C.POINTER(C.c_float)),C.POINTER(C.c_size_t),
                                  C.c_int,C.POINTER(C.c_float),C.c_size_t,C.POINTER(C.c_double)]
        self.handle=self.lib.bpu_create(str(model).encode())
        if not self.handle: raise RuntimeError(self.lib.bpu_last_error().decode())
        try:
            self.shapes=shapes
            for i, expected in enumerate(shapes):
                shape=(C.c_int*4)()
                self.check(self.lib.bpu_shape(self.handle,i,0,shape))
                assert tuple(shape)==tuple(expected),(tuple(shape),expected)
            shape=(C.c_int*4)()
            self.check(self.lib.bpu_shape(self.handle,0,1,shape))
            self.output_shape=tuple(shape)
        except Exception:
            self.close();raise

    def check(self, code):
        if code: raise RuntimeError(self.lib.bpu_last_error().decode())

    def run(self, arrays):
        assert len(arrays)==len(self.shapes)
        arrays=[np.ascontiguousarray(a,dtype=np.float32) for a in arrays]
        for array,shape in zip(arrays,self.shapes): assert array.shape==tuple(shape),(array.shape,shape)
        pointers=(C.POINTER(C.c_float)*len(arrays))(*[a.ctypes.data_as(C.POINTER(C.c_float)) for a in arrays])
        sizes=(C.c_size_t*len(arrays))(*[a.nbytes for a in arrays])
        output=np.empty(self.output_shape,np.float32)
        ms=C.c_double()
        self.check(self.lib.bpu_run(self.handle,pointers,sizes,len(arrays),
            output.ctypes.data_as(C.POINTER(C.c_float)),output.nbytes,C.byref(ms)))
        assert np.isfinite(output).all()
        return output,ms.value

    def close(self):
        if getattr(self,'handle',None):
            self.lib.bpu_close(self.handle);self.handle=None
