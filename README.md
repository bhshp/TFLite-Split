# README

Use
```bash
 flatc --cpp ./data/tflite.fbs --cpp-std c++17 --gen-mutable --gen-object-api --scoped-enums --cpp-ptr-type std::shared_ptr -o ./include/tflite_generated.hpp
```
to generate include files in `include/tflite_generated.hpp`

