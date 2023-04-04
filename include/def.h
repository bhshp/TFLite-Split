#pragma once

#include "tflite_generated.hpp"

#include <filesystem>   // std::filesystem
#include <memory>       // std::shared_ptr
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

template <typename T>
using PtrType = std::shared_ptr<T>;

template <typename T>
using RawPtrType = T*;

using RawDataType = PtrType<char[]>;

template <typename T>
using PtrContainerType = std::vector<PtrType<T>>;

template <typename T, typename... Args>
PtrType<T> make_ptr(Args&&... args) {
  if constexpr (std::is_same_v<std::unique_ptr<T>, PtrType<T>>) {
    return std::make_unique<T>(std::forward<Args&&>(args)...);
  }
  return std::make_shared<T>(std::forward<Args&&>(args)...);
}

namespace fs = std::filesystem;
