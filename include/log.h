#pragma once

#include <fmt/color.h>

#include <cstdio>   // std::putc
#include <cstdlib>  // std::abort
#include <string>   // std::string
#include <utility>  // std::forward std::unreachable

enum struct LogLevel { INFO, WARNING, ERROR, FATAL };

namespace detail {
fmt::text_style get_log_style(LogLevel level) {
  switch (level) {
    case LogLevel::INFO: {
      return fg(fmt::color::white);
    }
    case LogLevel::WARNING: {
      return fg(fmt::color::yellow);
    }
    case LogLevel::ERROR: {
      return fg(fmt::color::green);
    }
    case LogLevel::FATAL: {
      return fg(fmt::color::red);
    }
    default: {
      std::unreachable();
    }
  }
}

std::string get_log_level_text(LogLevel level) {
  switch (level) {
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::FATAL:
      return "FATAL";
    default:
      std::unreachable();
  }
}

template <typename... Args>
void log(LogLevel level, const std::string& fmt_str, Args&&... args) {
  const fmt::text_style& style = get_log_style(level);
  fmt::print(stderr, style, "[{}]: ", get_log_level_text(level));
  fmt::print(stderr, style, fmt_str, args...);
  std::putc('\n', stderr);
}

}  // namespace detail

template <typename... Args>
void log_info(const std::string& fmt_str, Args&&... args) {
  detail::log(LogLevel::INFO, fmt_str, std::forward<Args&&>(args)...);
}

template <typename... Args>
void log_warning(const std::string& fmt_str, Args&&... args) {
  detail::log(LogLevel::WARNING, fmt_str, std::forward<Args&&>(args)...);
}

template <typename... Args>
void log_error(const std::string& fmt_str, Args&&... args) {
  detail::log(LogLevel::ERROR, fmt_str, std::forward<Args&&>(args)...);
}

template <typename... Args>
void log_fatal(const std::string& fmt_str, Args&&... args) {
  detail::log(LogLevel::FATAL, fmt_str, std::forward<Args&&>(args)...);
  std::abort();
}
