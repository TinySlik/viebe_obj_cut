# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.9.3_1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.9.3_1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/tiny/Desktop/develop/vibe_custom

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/tiny/Desktop/develop/vibe_custom

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tiny/Desktop/develop/vibe_custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.o"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.o -c /Users/tiny/Desktop/develop/vibe_custom/main.cpp

CMakeFiles/main.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.i"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tiny/Desktop/develop/vibe_custom/main.cpp > CMakeFiles/main.dir/main.i

CMakeFiles/main.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.s"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tiny/Desktop/develop/vibe_custom/main.cpp -o CMakeFiles/main.dir/main.s

CMakeFiles/main.dir/main.o.requires:

.PHONY : CMakeFiles/main.dir/main.o.requires

CMakeFiles/main.dir/main.o.provides: CMakeFiles/main.dir/main.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.o.provides.build
.PHONY : CMakeFiles/main.dir/main.o.provides

CMakeFiles/main.dir/main.o.provides.build: CMakeFiles/main.dir/main.o


CMakeFiles/main.dir/myVibe.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/myVibe.o: myVibe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tiny/Desktop/develop/vibe_custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/myVibe.o"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/myVibe.o -c /Users/tiny/Desktop/develop/vibe_custom/myVibe.cpp

CMakeFiles/main.dir/myVibe.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/myVibe.i"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/tiny/Desktop/develop/vibe_custom/myVibe.cpp > CMakeFiles/main.dir/myVibe.i

CMakeFiles/main.dir/myVibe.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/myVibe.s"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/tiny/Desktop/develop/vibe_custom/myVibe.cpp -o CMakeFiles/main.dir/myVibe.s

CMakeFiles/main.dir/myVibe.o.requires:

.PHONY : CMakeFiles/main.dir/myVibe.o.requires

CMakeFiles/main.dir/myVibe.o.provides: CMakeFiles/main.dir/myVibe.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/myVibe.o.provides.build
.PHONY : CMakeFiles/main.dir/myVibe.o.provides

CMakeFiles/main.dir/myVibe.o.provides.build: CMakeFiles/main.dir/myVibe.o


CMakeFiles/main.dir/vibe-background-sequential.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/vibe-background-sequential.o: vibe-background-sequential.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/tiny/Desktop/develop/vibe_custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/main.dir/vibe-background-sequential.o"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/main.dir/vibe-background-sequential.o   -c /Users/tiny/Desktop/develop/vibe_custom/vibe-background-sequential.c

CMakeFiles/main.dir/vibe-background-sequential.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/main.dir/vibe-background-sequential.i"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/tiny/Desktop/develop/vibe_custom/vibe-background-sequential.c > CMakeFiles/main.dir/vibe-background-sequential.i

CMakeFiles/main.dir/vibe-background-sequential.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/main.dir/vibe-background-sequential.s"
	/Applications/developmentTools/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/tiny/Desktop/develop/vibe_custom/vibe-background-sequential.c -o CMakeFiles/main.dir/vibe-background-sequential.s

CMakeFiles/main.dir/vibe-background-sequential.o.requires:

.PHONY : CMakeFiles/main.dir/vibe-background-sequential.o.requires

CMakeFiles/main.dir/vibe-background-sequential.o.provides: CMakeFiles/main.dir/vibe-background-sequential.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/vibe-background-sequential.o.provides.build
.PHONY : CMakeFiles/main.dir/vibe-background-sequential.o.provides

CMakeFiles/main.dir/vibe-background-sequential.o.provides.build: CMakeFiles/main.dir/vibe-background-sequential.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.o" \
"CMakeFiles/main.dir/myVibe.o" \
"CMakeFiles/main.dir/vibe-background-sequential.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.o
main: CMakeFiles/main.dir/myVibe.o
main: CMakeFiles/main.dir/vibe-background-sequential.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libopencv_dnn.3.3.0.dylib
main: /usr/local/lib/libopencv_ml.3.3.0.dylib
main: /usr/local/lib/libopencv_objdetect.3.3.0.dylib
main: /usr/local/lib/libopencv_shape.3.3.0.dylib
main: /usr/local/lib/libopencv_stitching.3.3.0.dylib
main: /usr/local/lib/libopencv_superres.3.3.0.dylib
main: /usr/local/lib/libopencv_videostab.3.3.0.dylib
main: /usr/local/lib/libopencv_calib3d.3.3.0.dylib
main: /usr/local/lib/libopencv_features2d.3.3.0.dylib
main: /usr/local/lib/libopencv_flann.3.3.0.dylib
main: /usr/local/lib/libopencv_highgui.3.3.0.dylib
main: /usr/local/lib/libopencv_photo.3.3.0.dylib
main: /usr/local/lib/libopencv_video.3.3.0.dylib
main: /usr/local/lib/libopencv_videoio.3.3.0.dylib
main: /usr/local/lib/libopencv_imgcodecs.3.3.0.dylib
main: /usr/local/lib/libopencv_imgproc.3.3.0.dylib
main: /usr/local/lib/libopencv_core.3.3.0.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/tiny/Desktop/develop/vibe_custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/main.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/myVibe.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/vibe-background-sequential.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/tiny/Desktop/develop/vibe_custom && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tiny/Desktop/develop/vibe_custom /Users/tiny/Desktop/develop/vibe_custom /Users/tiny/Desktop/develop/vibe_custom /Users/tiny/Desktop/develop/vibe_custom /Users/tiny/Desktop/develop/vibe_custom/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

