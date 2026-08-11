# generate ninja makefiles (defaults to Debug)
cmake -S . -B build -G Ninja -DENABLE_TESTING=OFF

# build using ninja
ninja -C build