cmake_minimum_required(VERSION 3.16)
# The portable engine component lives in this project's components/ dir.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(whistle_demo)
