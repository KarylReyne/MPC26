# this should work for all CMake-based projects
mkdir build
cd build
cmake ..
make
<!-- ./executable -->

<!-- run executable with memory check -->
compute-sanitizer --tool memcheck ./executable [executable options]