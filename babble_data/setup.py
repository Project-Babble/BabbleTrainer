from setuptools import setup, Extension
import numpy
import os
import sys
import subprocess

# --- OpenCV Configuration ---

# Set this environment variable to point to your OpenCV build directory
# e.g., C:\opencv\build or /usr/local
opencv_dir = "C:\\opencv\\build\\"#os.environ.get('OPENCV_DIR')

include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # --- Windows Configuration ---
    if not opencv_dir:
        raise ValueError("Please set the OPENCV_DIR environment variable to your OpenCV build directory.")
    
    include_dirs.append(os.path.join(opencv_dir, 'include'))
    # Adjust 'vc15' or 'vc16' based on your Visual Studio version
    library_dirs.append(os.path.join(opencv_dir, 'x64', 'vc16', 'lib'))
    
    # Find the correct opencv_world library file
    # For example, opencv_world455.lib for version 4.5.5
    # You may need to update this version number
    libraries.append('opencv_world4100') 

else:
    # --- Linux/macOS Configuration using pkg-config ---
    try:
        # This is the most reliable way on Linux/macOS
        cflags = subprocess.check_output(['pkg-config', 'opencv4', '--cflags']).decode('utf-8').strip().split()
        libs = subprocess.check_output(['pkg-config', 'opencv4', '--libs']).decode('utf-8').strip().split()
        
        for flag in cflags:
            if flag.startswith('-I'):
                include_dirs.append(flag[2:])
        
        for flag in libs:
            if flag.startswith('-L'):
                library_dirs.append(flag[2:])
            elif flag.startswith('-l'):
                libraries.append(flag[2:])
            else:
                extra_link_args.append(flag)

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: pkg-config opencv4 not found. Falling back to OPENCV_DIR.")
        if not opencv_dir:
            raise ValueError("pkg-config failed and OPENCV_DIR is not set. Please configure OpenCV.")
        
        include_dirs.append(os.path.join(opencv_dir, 'include', 'opencv4'))
        library_dirs.append(os.path.join(opencv_dir, 'lib'))
        libraries.extend(['opencv_core', 'opencv_imgproc'])


# --- Define the Extension ---
babble_data_extension = Extension(
    'babble_data',
    sources=['babble_data.cpp'], # IMPORTANT: changed to .cpp
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++' # Tell setuptools to use the C++ compiler
)

setup(
    name='babble_data',
    version='1.3',
    description='A C++ extension using OpenCV for NumPy processing.',
    ext_modules=[babble_data_extension]
)