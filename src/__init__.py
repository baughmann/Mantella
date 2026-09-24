import os
import sys

# pandas 1.5.3 ships its own, older msvcp140.dll (pandas/_libs/window). Whichever copy loads
# first is reused by every later DLL that asks for msvcp140.dll by name, and newer native wheels
# (onnxruntime >= 1.21, torch) crash or fail DLL init against the old one. Load the system copy
# (installed by the VC++ redistributable) before anything imports pandas.
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.WinDLL(os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'msvcp140.dll'))
    except OSError:
        pass
