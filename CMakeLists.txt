   project( VIBE_EX )

   find_package(Threads)
   target_link_libraries(main ${CMAKE_THREAD_LIBS_INIT})
   find_package( OpenCV REQUIRED )
   add_executable(main main.cpp EXTfunction.cpp  TODynamiacBackgroundExtraction.cpp vibe-background-sequential.c)
   target_link_libraries( main ${OpenCV_LIBS} )
