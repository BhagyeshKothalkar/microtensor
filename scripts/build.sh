# generate ninja makefiles (defaults to Debug)
cmake -S . -B build -G Ninja

# build using ninja
ninja -C build