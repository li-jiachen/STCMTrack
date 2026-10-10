"""https://github.com/benhoyt/namedmutex
Named mutex handling (for Win32).
See README.md or https://github.com/benhoyt/namedmutex for a bit more
documentation.
This code is released under the new BSD 3-clause license:
http://opensource.org/licenses/BSD-3-Clause
"""

import ctypes
from ctypes import wintypes


_CreateMutex = ctypes.windll.kernel32.CreateMutexW
_CreateMutex.argtypes = [wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCWSTR]
_CreateMutex.restype = wintypes.HANDLE

_WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
_WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_WaitForSingleObject.restype = wintypes.DWORD

_ReleaseMutex = ctypes.windll.kernel32.ReleaseMutex
_ReleaseMutex.argtypes = [wintypes.HANDLE]
_ReleaseMutex.restype = wintypes.BOOL

_CloseHandle = ctypes.windll.kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL


class NamedMutex(object):
    

    def __init__(self, name, acquired=False):
        
        self.name = name
        self.acquired = acquired
        ret = _CreateMutex(None, False, name)
        if not ret:
            raise ctypes.WinError()
        self.handle = ret
        if acquired:
            self.acquire()

    def acquire(self, timeout=None):
        
        if timeout is None:
            
            timeout = 0xFFFFFFFF
        else:
            timeout = int(round(timeout * 1000))
        ret = _WaitForSingleObject(self.handle, timeout)
        if ret in (0, 0x80):
            
            
            self.acquired = True
            return True
        elif ret == 0x102:
            
            self.acquired = False
            return False
        else:
            
            raise ctypes.WinError()

    def release(self):
        
        ret = _ReleaseMutex(self.handle)
        if not ret:
            raise ctypes.WinError()
        self.acquired = False

    def close(self):
        
        if self.handle is None:
            
            return
        ret = _CloseHandle(self.handle)
        if not ret:
            raise ctypes.WinError()
        self.handle = None

    __del__ = close

    def __repr__(self):
        
        return '{0}({1!r}, acquired={2})'.format(
            self.__class__.__name__, self.name, self.acquired)

    __str__ = __repr__

    
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if __name__ == '__main__':
    
    with NamedMutex('test_mutex_123'):
        pass