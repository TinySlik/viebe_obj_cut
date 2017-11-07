   project( VIBE_EX )
   find_package( OpenCV REQUIRED )
   add_executable(main main.cpp EXTfunction.cpp  TODynamiacBackgroundExtraction.cpp vibe-background-sequential.c)
   target_link_libraries( main ${OpenCV_LIBS} )
