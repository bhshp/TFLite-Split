#pragma once

#include <compare>
#include <vector>

template <typename T>
  requires std::three_way_comparable<T>
void deduplicate(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}
