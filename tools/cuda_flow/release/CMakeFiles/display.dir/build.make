# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /files/vol07/LSEL/dev/linux/cmake/3.1.0/bin/cmake

# The command to remove a file.
RM = /files/vol07/LSEL/dev/linux/cmake/3.1.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release

# Include any dependencies generated for this target.
include CMakeFiles/display.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/display.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/display.dir/flags.make

CMakeFiles/display.dir/display.o: CMakeFiles/display.dir/flags.make
CMakeFiles/display.dir/display.o: ../display.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/display.dir/display.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/display.dir/display.o -c /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/display.cpp

CMakeFiles/display.dir/display.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/display.dir/display.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/display.cpp > CMakeFiles/display.dir/display.i

CMakeFiles/display.dir/display.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/display.dir/display.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/display.cpp -o CMakeFiles/display.dir/display.s

CMakeFiles/display.dir/display.o.requires:
.PHONY : CMakeFiles/display.dir/display.o.requires

CMakeFiles/display.dir/display.o.provides: CMakeFiles/display.dir/display.o.requires
	$(MAKE) -f CMakeFiles/display.dir/build.make CMakeFiles/display.dir/display.o.provides.build
.PHONY : CMakeFiles/display.dir/display.o.provides

CMakeFiles/display.dir/display.o.provides.build: CMakeFiles/display.dir/display.o

CMakeFiles/display.dir/flow_functions.o: CMakeFiles/display.dir/flags.make
CMakeFiles/display.dir/flow_functions.o: ../flow_functions.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/display.dir/flow_functions.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/display.dir/flow_functions.o -c /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/flow_functions.cpp

CMakeFiles/display.dir/flow_functions.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/display.dir/flow_functions.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/flow_functions.cpp > CMakeFiles/display.dir/flow_functions.i

CMakeFiles/display.dir/flow_functions.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/display.dir/flow_functions.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/flow_functions.cpp -o CMakeFiles/display.dir/flow_functions.s

CMakeFiles/display.dir/flow_functions.o.requires:
.PHONY : CMakeFiles/display.dir/flow_functions.o.requires

CMakeFiles/display.dir/flow_functions.o.provides: CMakeFiles/display.dir/flow_functions.o.requires
	$(MAKE) -f CMakeFiles/display.dir/build.make CMakeFiles/display.dir/flow_functions.o.provides.build
.PHONY : CMakeFiles/display.dir/flow_functions.o.provides

CMakeFiles/display.dir/flow_functions.o.provides.build: CMakeFiles/display.dir/flow_functions.o

# Object files for target display
display_OBJECTS = \
"CMakeFiles/display.dir/display.o" \
"CMakeFiles/display.dir/flow_functions.o"

# External object files for target display
display_EXTERNAL_OBJECTS =

display: CMakeFiles/display.dir/display.o
display: CMakeFiles/display.dir/flow_functions.o
display: CMakeFiles/display.dir/build.make
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_videostab.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_ts.a
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_superres.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_stitching.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_contrib.so.2.4.10
display: /usr/lib64/libGLU.so
display: /usr/lib64/libGL.so
display: /usr/lib64/libSM.so
display: /usr/lib64/libICE.so
display: /usr/lib64/libX11.so
display: /usr/lib64/libXext.so
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_nonfree.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_ocl.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_gpu.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_photo.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_objdetect.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_legacy.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_video.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_ml.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_calib3d.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_features2d.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_highgui.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_imgproc.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_flann.so.2.4.10
display: /files/vol07/LSEL/dev/linux/opencv/2.4.10.1/lib/libopencv_core.so.2.4.10
display: CMakeFiles/display.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable display"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/display.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/display.dir/build: display
.PHONY : CMakeFiles/display.dir/build

CMakeFiles/display.dir/requires: CMakeFiles/display.dir/display.o.requires
CMakeFiles/display.dir/requires: CMakeFiles/display.dir/flow_functions.o.requires
.PHONY : CMakeFiles/display.dir/requires

CMakeFiles/display.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/display.dir/cmake_clean.cmake
.PHONY : CMakeFiles/display.dir/clean

CMakeFiles/display.dir/depend:
	cd /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release /home/ninaoa1/Projects/temp/lstm/lstm7/lisa-caffe-public/tools/cuda_flow/release/CMakeFiles/display.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/display.dir/depend

