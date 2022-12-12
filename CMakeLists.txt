# CMakeList.txt : CMake project for CGProj, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("CGProj")

# Add source to this project's executable.
add_executable (CGProj "CGProj.cpp"  "hdr.h" "hdr.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CGProj PROPERTY CXX_STANDARD 20)
endif()


# Set latest C++ standard
if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
    endif()
endif()

# TODO: Add tests and install targets if needed.
