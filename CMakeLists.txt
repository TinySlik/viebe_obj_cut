   project( VIBE_EX )
   find_package( OpenCV REQUIRED )
   add_executable(main main.cpp myVibe.cpp vibe-background-sequential.c)
   target_link_libraries( main ${OpenCV_LIBS} )
