#pragma once

#include <fmt/core.h>

#include <compare>
#include <string>
#include <string_view>
#include <vector>

template <typename T>
  requires std::three_way_comparable<T>
void deduplicate(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

template <typename Range>
std::string join(Range&& range, std::string_view sep) {
  return fmt::format("{}", fmt::join(std::forward<Range&&>(range), sep));
}
