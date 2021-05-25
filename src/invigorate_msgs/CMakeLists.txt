cmake_minimum_required(VERSION 2.8.3)
project(invigorate_msgs)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Boost REQUIRED thread)

include_directories(SYSTEM "msg_gen/cpp/include")

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
  sensor_msgs
  rospy
)

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

# add_message_files(
#   FILES
# )

add_service_files(
  FILES
  # DenseCaption.srv
  # MeteorScore.srv
  Grounding.srv
  MAttNetGrounding.srv
  ObjectDetection.srv
  VmrDetection.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  DEPENDS
  CATKIN_DEPENDS message_runtime
  sensor_msgs
)

include_directories(include)

# # add_library(action_controller src/actionController.cpp)
# add_executable(action_controller src/actionController.cpp)
# target_link_libraries(action_controller ${catkin_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

#install(TARGETS action_controller LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})


