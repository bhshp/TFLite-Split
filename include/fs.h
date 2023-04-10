#pragma once

#include <cassert>        // assert
#include <cstddef>        // size_t
#include <fstream>        // std::ifstream std::ios::binary
#include <functional>     // std::function
#include <ranges>         // std::cartesian_product
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::pair std::make_pair
#include <vector>         // std::vector

#include "def.h"
#include "log.h"
#include "tflite_generated.hpp"
#include "utility.h"

std::pair<RawDataType, size_t> read_binary_from_path(fs::path file_path) {
  file_path = fs::canonical(file_path);
  RawDataType data = nullptr;
  size_t size = 0;
  if (file_path.extension() != ".tflite") {
    log_fatal("File format not correct: {}, but we need .tflite.",
              file_path.extension().string());
  } else if (!fs::exists(file_path) || fs::is_directory(file_path)) {
    log_fatal("File {} does not exist or is a directory.", file_path.c_str());
  } else {
    std::ifstream input{file_path, std::ios::binary};
    size = fs::file_size(file_path);
    const static size_t mb_in_byte = 1024 * 1024;
    log_info("Opening file {} of {} Bytes ({} MB).",
             file_path.c_str(),
             size,
             (size + mb_in_byte - 1) / mb_in_byte);
    data = std::make_shared<char[]>(size);
    input.read(data.get(), size);
  }
  return std::make_pair(data, size);
}

void save_as_tflite(fs::path file_path, const tflite::ModelT& model_table) {
  if (!file_path.has_extension() || file_path.extension() != ".tflite") {
    file_path.replace_extension(".tflite");
    log_warning(
        "Path of output is not in tflite format, replace it with tflite "
        "format: {}.",
        file_path.string());
  }

  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(tflite::CreateModel(builder, &model_table),
                 tflite::ModelIdentifier());

  const uint8_t* saved_data = builder.GetBufferPointer();
  size_t saved_size = builder.GetSize();

  fs::remove_all(file_path);

  std::ofstream output(file_path, std::ios::binary | std::ios::out);
  output.write(reinterpret_cast<const char*>(saved_data), saved_size);
}

void save_summary(const tflite::ModelT& model_table,
                  fs::path model_name,
                  fs::path model_folder) {
  fs::path summary_path = model_folder / model_name.replace_extension(".txt");
  fs::remove_all(summary_path);
  std::ofstream os(summary_path);

  os << model_name << std::endl;
  os << model_table.subgraphs.size() << std::endl;  // print subgraph numbers
  for (PtrType<tflite::SubGraphT> subgraph_ptr : model_table.subgraphs) {
    int n = subgraph_ptr->tensors.size();

    os << n << std::endl;  // print tensor numbers
    for (PtrType<tflite::TensorT> tensor_ptr : subgraph_ptr->tensors) {
      os << tensor_ptr->name << '\n'
         << tflite::EnumNameTensorType(tensor_ptr->type) << '\t'
         << tensor_ptr->buffer << '\t'
         << model_table.buffers[tensor_ptr->buffer]->data.size() << '\t'
         << join(tensor_ptr->shape_signature, " ")
         << std::endl;
    }

    os << subgraph_ptr->operators.size() << std::endl;
    for (PtrType<tflite::OperatorT> operator_ptr : subgraph_ptr->operators) {
      std::vector<int32_t> valid_inputs = operator_ptr->inputs,
                           valid_outputs = operator_ptr->outputs;
      os << '#' << std::endl;
      os << join(valid_inputs, " ") << std::endl;
      os << join(valid_outputs, " ") << std::endl;
    }
  }
}

void save_operator(fs::path save_path,
                   const tflite::ModelT& model_table,
                   PtrType<tflite::SubGraphT> subgraph_ptr,
                   PtrType<tflite::OperatorT> op_ptr) {
  PtrType<tflite::OperatorT> new_op = make_ptr<tflite::OperatorT>();
  new_op->opcode_index = op_ptr->opcode_index;
  new_op->builtin_options = op_ptr->builtin_options;
  new_op->custom_options = op_ptr->custom_options;
  new_op->mutating_variable_inputs = op_ptr->mutating_variable_inputs;
  new_op->intermediates = op_ptr->intermediates;

  std::vector<int32_t> temp_indices;
  std::unordered_map<int32_t, int32_t> tensor_indices_map;
  {
    temp_indices.reserve(op_ptr->inputs.size() + op_ptr->outputs.size());
    temp_indices.insert(
        temp_indices.end(), op_ptr->inputs.begin(), op_ptr->inputs.end());
    temp_indices.insert(
        temp_indices.end(), op_ptr->outputs.begin(), op_ptr->outputs.end());

    deduplicate(temp_indices);

    int32_t index = 0;
    for (int x : temp_indices) {
      tensor_indices_map.try_emplace(x, index++);
    }
  }

  new_op->inputs.reserve(op_ptr->inputs.size());
  for (int32_t input_index : op_ptr->inputs) {
    new_op->inputs.emplace_back(tensor_indices_map[input_index]);
  }

  new_op->outputs.reserve(op_ptr->outputs.size());
  for (int32_t output_index : op_ptr->outputs) {
    new_op->outputs.emplace_back(tensor_indices_map[output_index]);
  }

  // bottom-up construction

  tflite::ModelT new_model_table;
  new_model_table.version = model_table.version;
  new_model_table.operator_codes = model_table.operator_codes;
  new_model_table.description = model_table.description;

  PtrType<tflite::SubGraphT>& new_subgraph =
      new_model_table.subgraphs.emplace_back(make_ptr<tflite::SubGraphT>());
  new_subgraph->tensors.reserve(temp_indices.size());

  std::vector<uint32_t> buffer_indices;
  std::unordered_map<uint32_t, uint32_t> buffer_indices_map;
  {
    buffer_indices.reserve(temp_indices.size());
    for (int32_t index : temp_indices) {
      PtrType<tflite::TensorT> new_tensor_ptr =
          new_subgraph->tensors.emplace_back(
              make_ptr<tflite::TensorT>(*subgraph_ptr->tensors[index]));
      buffer_indices.emplace_back(new_tensor_ptr->buffer);
    }
    buffer_indices.emplace_back(0);
    deduplicate(buffer_indices);
    uint32_t index = 0;
    for (int x : buffer_indices) {
      buffer_indices_map.try_emplace(x, index++);
    }
  }

  new_model_table.buffers.reserve(buffer_indices.size());
  for (int buffer_index : buffer_indices) {
    new_model_table.buffers.emplace_back(model_table.buffers[buffer_index]);
  }

  for (PtrType<tflite::TensorT> new_tensor_ptr : new_subgraph->tensors) {
    new_tensor_ptr->buffer = buffer_indices_map[new_tensor_ptr->buffer];
  }

  std::function<bool(int32_t)> is_invalid_tensor = [&](int32_t x) -> bool {
    return new_subgraph->tensors[x]->shape_signature.empty();
  };

  new_subgraph->inputs = new_op->inputs;
  std::erase_if(new_subgraph->inputs, is_invalid_tensor);

  new_subgraph->outputs = new_op->outputs;
  std::erase_if(new_subgraph->outputs, is_invalid_tensor);

  new_subgraph->operators = {new_op};
  new_subgraph->name = subgraph_ptr->name;

  save_as_tflite(save_path, new_model_table);
}

void save_operators(const tflite::ModelT& model_table,
                    fs::path model_name,
                    fs::path root_folder) {
  if (fs::exists(root_folder) && !fs::is_directory(root_folder)) {
    log_fatal("{} exists and is not a folder, abort.", root_folder.string());
    return;
  } else {
    log_warning("Creating {} as if it does not exist.", root_folder.string());
    fs::create_directories(root_folder);
  }

  fs::path model_folder = root_folder / model_name;
  if (fs::exists(model_folder) && !fs::is_directory(model_folder)) {
    log_fatal("{} exists and is not a folder, abort.", model_folder.string());
  } else {
    log_warning("Cleaning all contents in {}.", model_folder.string());
    fs::remove_all(model_folder);
    fs::create_directories(model_folder);
  }

  save_summary(model_table, model_name, model_folder);

  for (size_t subgraph_index = 0, N = model_table.subgraphs.size();
       subgraph_index < N;
       ++subgraph_index) {
    PtrType<tflite::SubGraphT> subgraph_ptr =
        model_table.subgraphs[subgraph_index];
    const PtrContainerType<tflite::OperatorT>& operators =
        subgraph_ptr->operators;
    for (size_t operator_index = 0, M = operators.size(); operator_index < M;
         ++operator_index) {
      fs::path save_path =
          model_folder / (model_name.string()
                              .append("_")
                              .append(std::to_string(subgraph_index))
                              .append("_")
                              .append(std::to_string(operator_index))
                              .append(".tflite"));
      save_operator(
          save_path, model_table, subgraph_ptr, operators[operator_index]);
    }
  }
}
