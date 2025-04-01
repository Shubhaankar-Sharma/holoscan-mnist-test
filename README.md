### build
```
mkdir build
cd build
cmake ..
make
```

### run with nsys profiling
```
nsys profile   --trace=cuda,nvtx,cudnn,cublas   --sample=none   --cuda-memory-usage=true   --stats=true   --force-overwrite=true   -o mnist_profile -d 3 ./build/mnist_app
```

