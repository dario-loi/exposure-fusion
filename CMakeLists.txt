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

# Add OpenMP for easy multi-threading of graphic routines.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CGProj PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(CGProj PROPERTIES
    VS_GLOBAL_RunCodeAnalysis true

    # Use visual studio core guidelines
    VS_GLOBAL_EnableMicrosoftCodeAnalysis true
    VS_GLOBAL_CodeAnalysisRuleSet ${CMAKE_CURRENT_SOURCE_DIR}/CGProj.ruleset
    VS_GLOBAL_CodeAnalysisRuleSet ${CMAKE_CURRENT_SOURCE_DIR}/CGProj.ruleset

    # Use clangtidy
    VS_GLOBAL_EnableClangTidyCodeAnalysis true
    VS_GLOBAL_ClangTidyChecks -checks=-*,modernize-*,-modernize-use-trailing-return-type
)

# TODO: Add tests and install targets if needed.
