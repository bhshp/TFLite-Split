#include <string>
#include <string_view>
#include <vector>

#include "argparse.hpp"
#include "fs.h"
#include "tflite_generated.hpp"

int main(int argc, char** argv) {

  const std::string_view input_flag = "--input_file";
  const std::string_view output_flag = "--output_root_folder";

  argparse::ArgumentParser parser("split_tflite");
  parser.add_argument(input_flag)
      .required()
      .help("Input file of tflite format");
  parser.add_argument(output_flag)
      .default_value(std::filesystem::current_path().string())
      .help("Root directory of output folder");
  std::vector<std::string> unknown_args = parser.parse_known_args(argc, argv);
  if (!unknown_args.empty()) {
    log_fatal("unknown args: [{}]", fmt::join(unknown_args, ", "));
  }

  log_warning("current path: {}", std::filesystem::current_path().string());
  std::filesystem::path file_path = parser.get<std::string>(input_flag);
  std::filesystem::path root_folder = parser.get<std::string>(output_flag);

  if (!parser.is_used(output_flag)) {
    log_warning("{} is unset, using default value {} now.",
                output_flag,
                root_folder.string());
  }

  auto [data, size] = read_binary_from_path(file_path);

  if (data == nullptr || size == 0) {
    return EXIT_FAILURE;
  }

  tflite::Model* model = tflite::GetMutableModel(data.get());

  tflite::ModelT model_table;
  model->UnPackTo(&model_table);

  std::filesystem::path model_name = file_path.stem();

  save_operators(model_table, model_name, root_folder);

  return EXIT_SUCCESS;
}
