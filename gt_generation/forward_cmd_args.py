# forward command line arguments to esmini
# using ideas from: https://comp.lang.python.narkive.com/RUdYlz64/trying-to-pass-sys-argv-as-int-argc-char-argv-using-ctypes

import os
import ctypes as ct
import sys

lib_path = os.path.join(os.path.expanduser('~'), "esmini")

lib_paths = {
    "linux": os.path.join(lib_path, "bin/libesminiLib.so"),
    "linux2": os.path.join(lib_path, "bin/libesminiLib.so"),
    "darwin": os.path.join(lib_path, "bin/libesminiLib.dylib"),
    "win32": os.path.join(lib_path, "esminiLib.dll"),
}
se = ct.CDLL(lib_paths[sys.platform])

# specify arguments types of esmini function
se.SE_InitWithArgs.argtypes = [ct.c_int, ct.POINTER(ct.POINTER(ct.c_char))]

# fetch command line arguments
argc = len(sys.argv)
argv = (ct.POINTER(ct.c_char) * (argc + 1))()
for i, arg in enumerate(sys.argv):
    argv[i] = ct.create_string_buffer(arg.encode('utf-8'))

# init esmini
if se.SE_InitWithArgs(argc, argv) != 0:
    exit(-1)

# execute esmini until end of scenario or user requested quit by ESC key
while se.SE_GetQuitFlag() == 0:
    se.SE_Step()