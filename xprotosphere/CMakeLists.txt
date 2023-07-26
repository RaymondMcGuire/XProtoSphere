# Project Name
set(PROJECT_NAME xprotosphere)

# Find Extra Library Package
set(EXTLIBS_INCLUDE ${CMAKE_SOURCE_DIR}/external/KiriExtlib/include)
set(EXTLIBS_SRC ${CMAKE_SOURCE_DIR}/external/KiriExtlib/src)
set(EXTLIBS_LIB ${CMAKE_SOURCE_DIR}/external/KiriExtlib/lib)
link_directories(${EXTLIBS_LIB})

if(WIN32)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/)
    message(STATUS "CMAKE_MODULE_PATH   : ${CMAKE_MODULE_PATH}")
    message(STATUS "Found ASSIMP in ${ASSIMP_INCLUDE_DIR}")
    find_package(OpenMP REQUIRED)

    if(OPENMP_FOUND)
        message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
        message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# PBS CUDA Library
set(XPROTOSPHERE_CUDA_LIB_INCLUDE ${CMAKE_BINARY_DIR}/xprotosphere_cuda/include)

# Kiri Math Library
set(KIRI_MATH_LIB_INCLUDE ${CMAKE_SOURCE_DIR}/external/KiriMathMini/include)

# Configuration File
set(BUILD_STRING "build")


configure_file(${CMAKE_SOURCE_DIR}/configuration/root_directory.h.in ${CMAKE_BINARY_DIR}/configuration/root_directory.h)
set(CONFIGURATION_INCLUDE ${CMAKE_BINARY_DIR}/configuration)

# Extra Static Library
set(EXTLIBS_LINK_LIBS_DEBUG Alembic xprotosphere_cuda kiri_math_mini)

# Macro Command
macro(makeLink src dest target)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest} DEPENDS ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

# Project Config
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")
set(PROJECT_FILES ${SOURCES} ${HEADERS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ../bin/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ../bin/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${EXTLIBS_INCLUDE}
    ${CONFIGURATION_INCLUDE}
    ${XPROTOSPHERE_CUDA_LIB_INCLUDE}
    ${KIRI_MATH_LIB_INCLUDE}
)

target_link_libraries(${PROJECT_NAME} ${EXTLIBS_LINK_LIBS_DEBUG} debug partiod optimized partio)

set(WD_DEBUG_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Debug/)
set(WD_RELEASE_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Release/)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME_DEBUG ${PROJECT_NAME}d
    OUTPUT_NAME_RELEASE ${PROJECT_NAME}
    VS_DEBUGGER_WORKING_DIRECTORY "$<$<CONFIG:debug>:${WD_DEBUG_FILE_PATH}>$<$<CONFIG:release>:${WD_RELEASE_FILE_PATH}>"
)

# Copy DLLs
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/external/KiriExtlib/dlls" $<TARGET_FILE_DIR:${PROJECT_NAME}>)
