
cmake_minimum_required(VERSION 3.12)
project(tinygizmo_example VERSION 0.1.0 LANGUAGES C CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
    set (PLATFORM_DEFS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# Because rpaths have to be as fiendishly difficult as possible on Mac:
if (APPLE)

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

endif()


set(TINYGIZMO_EXAMPLE_SRC
    example-main.cpp
    gl-api.hpp
    teapot.h
    util.hpp
)

if(APPLE)
    set(TINYGIZMO_DEFS
    )
else()
    set(TINYGIZMO_DEFS
    )
endif()

find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)

#-------------------------------------------------------------------------------
# tinygizmo
#-------------------------------------------------------------------------------

if (WIN32)
endif()

set(TINYGIZMO_SRC
    ../tiny-gizmo.cpp
    ../tiny-gizmo.hpp
)

add_library(tinygizmo STATIC ${TINYGIZMO_SRC})
target_include_directories(tinygizmo SYSTEM
    PUBLIC ..)
target_compile_definitions(tinygizmo PRIVATE
    ${PLATFORM_DEFS}
)

#-------------------------------------------------------------------------------
# tinygizmo_example
#-------------------------------------------------------------------------------

add_executable(tinygizmo_example
    ${TINYGIZMO_EXAMPLE_SRC}
)

set_target_properties(tinygizmo_example PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY bin)

target_compile_definitions(tinygizmo_example PRIVATE
    ${PLATFORM_DEFS}
)

target_include_directories(tinygizmo_example
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${GLEW_INCLUDE_DIR}
    )

if (WIN32)
    set(PLATFORM_LIBS ws2_32 Iphlpapi.lib opengl32.lib)
elseif (APPLE)
#    if(CMAKE_OSX_SYSROOT MATCHES ".*iphoneos.*")
#        set(DARWIN_LIBS
#            "-framework AudioToolbox"
#            "-framework Accelerate"
#            "-framework CoreAudio")
#    else()
        set(PLATFORM_LIBS
            "-framework AudioToolbox"
            "-framework AudioUnit"
            "-framework Accelerate"
            "-framework Cocoa"
            "-framework CoreAudio"
            "-framework Metal"
            "-framework MetalKit"
            "-framework QuartzCore"
            "-framework OpenGL"
            )
#    endif()
endif()

target_link_libraries(tinygizmo_example
    ${PLATFORM_LIBS} ${GLEW_SHARED_LIBRARY_RELEASE} glfw tinygizmo
    )

#-------------------------------------------------------------------------------
# Installer
#-------------------------------------------------------------------------------

install(
    TARGETS tinygizmo_example
    BUNDLE DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

