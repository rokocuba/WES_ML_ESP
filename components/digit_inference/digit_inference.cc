#include "./digit_inference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "mnist_tiny_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr const char* kTag = "digit_inference";
constexpr int kResolverOps = 16;

const tflite::Model* s_model = nullptr;
tflite::MicroInterpreter* s_interpreter = nullptr;
TfLiteTensor* s_input = nullptr;
TfLiteTensor* s_output = nullptr;

tflite::MicroErrorReporter s_error_reporter;
tflite::MicroMutableOpResolver<kResolverOps> s_resolver;
bool s_resolver_ready = false;

alignas(tflite::MicroInterpreter) uint8_t
    s_interpreter_storage[sizeof(tflite::MicroInterpreter)];
uint8_t* s_tensor_arena = nullptr;
size_t s_tensor_arena_size = 0;

bool add_resolver_op(TfLiteStatus status, const char* name) {
  if (status == kTfLiteOk) {
    return true;
  }
  ESP_LOGE(kTag, "Failed to register op: %s", name);
  return false;
}

bool setup_resolver_once() {
  if (s_resolver_ready) {
    return true;
  }

  if (!add_resolver_op(s_resolver.AddConv2D(), "Conv2D")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddMaxPool2D(), "MaxPool2D")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddAveragePool2D(), "AveragePool2D")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddFullyConnected(), "FullyConnected")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddReshape(), "Reshape")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddShape(), "Shape")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddStridedSlice(), "StridedSlice")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddPack(), "Pack")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddSoftmax(), "Softmax")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddQuantize(), "Quantize")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddDequantize(), "Dequantize")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddAdd(), "Add")) {
    return false;
  }
  if (!add_resolver_op(s_resolver.AddMul(), "Mul")) {
    return false;
  }

  s_resolver_ready = true;
  return true;
}

void free_tensor_arena() {
  if (s_tensor_arena != nullptr) {
    heap_caps_free(s_tensor_arena);
    s_tensor_arena = nullptr;
    s_tensor_arena_size = 0;
  }
}

bool allocate_tensor_arena(size_t arena_size) {
#if CONFIG_DIGIT_INFERENCE_ARENA_IN_SPIRAM
  s_tensor_arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(
      16, arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (s_tensor_arena != nullptr) {
    s_tensor_arena_size = arena_size;
    ESP_LOGI(kTag, "Tensor arena allocated in PSRAM (%u bytes)",
             (unsigned)arena_size);
    return true;
  }
#endif

  s_tensor_arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(
      16, arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (s_tensor_arena != nullptr) {
    s_tensor_arena_size = arena_size;
    ESP_LOGI(kTag, "Tensor arena allocated in internal RAM (%u bytes)",
             (unsigned)arena_size);
    return true;
  }

  s_tensor_arena = static_cast<uint8_t*>(
      heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_8BIT));
  if (s_tensor_arena != nullptr) {
    s_tensor_arena_size = arena_size;
    ESP_LOGI(kTag, "Tensor arena allocated with generic heap (%u bytes)",
             (unsigned)arena_size);
    return true;
  }

  return false;
}

esp_err_t quantize_or_copy_input(
    const uint8_t input_28x28[DIGIT_INFERENCE_INPUT_SIZE]) {
  if (s_input == nullptr) {
    return ESP_ERR_INVALID_STATE;
  }

  if (s_input->type == kTfLiteInt8) {
    const float scale = s_input->params.scale;
    const int zero_point = s_input->params.zero_point;
    if (scale == 0.0f) {
      return ESP_ERR_INVALID_STATE;
    }

    int8_t* dst = s_input->data.int8;
    for (int i = 0; i < DIGIT_INFERENCE_INPUT_SIZE; ++i) {
      const float x = static_cast<float>(input_28x28[i]) / 255.0f;
      int q = static_cast<int>(lroundf(x / scale)) + zero_point;
      q = std::max(-128, std::min(127, q));
      dst[i] = static_cast<int8_t>(q);
    }
    return ESP_OK;
  }

  if (s_input->type == kTfLiteFloat32) {
    float* dst = s_input->data.f;
    for (int i = 0; i < DIGIT_INFERENCE_INPUT_SIZE; ++i) {
      dst[i] = static_cast<float>(input_28x28[i]) / 255.0f;
    }
    return ESP_OK;
  }

  ESP_LOGE(kTag, "Unsupported input tensor type: %d",
           static_cast<int>(s_input->type));
  return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t read_output_probabilities(float probs[DIGIT_INFERENCE_NUM_CLASSES]) {
  if (s_output == nullptr) {
    return ESP_ERR_INVALID_STATE;
  }

  float logits[DIGIT_INFERENCE_NUM_CLASSES];
  memset(logits, 0, sizeof(logits));

  if (s_output->type == kTfLiteInt8) {
    const float scale = s_output->params.scale;
    const int zero_point = s_output->params.zero_point;
    if (scale == 0.0f) {
      return ESP_ERR_INVALID_STATE;
    }

    const int n = std::min(DIGIT_INFERENCE_NUM_CLASSES,
                           static_cast<int>(s_output->bytes));
    for (int i = 0; i < n; ++i) {
      logits[i] = scale * (static_cast<float>(s_output->data.int8[i]) -
                           static_cast<float>(zero_point));
    }
  } else if (s_output->type == kTfLiteFloat32) {
    const int n = std::min(DIGIT_INFERENCE_NUM_CLASSES,
                           static_cast<int>(s_output->bytes / sizeof(float)));
    for (int i = 0; i < n; ++i) {
      logits[i] = s_output->data.f[i];
    }
  } else {
    ESP_LOGE(kTag, "Unsupported output tensor type: %d",
             static_cast<int>(s_output->type));
    return ESP_ERR_NOT_SUPPORTED;
  }

  float max_logit = logits[0];
  for (int i = 1; i < DIGIT_INFERENCE_NUM_CLASSES; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < DIGIT_INFERENCE_NUM_CLASSES; ++i) {
    probs[i] = expf(logits[i] - max_logit);
    sum += probs[i];
  }

  if (sum <= 0.0f) {
    return ESP_ERR_INVALID_STATE;
  }

  for (int i = 0; i < DIGIT_INFERENCE_NUM_CLASSES; ++i) {
    probs[i] /= sum;
  }

  return ESP_OK;
}

}  // namespace

extern "C" esp_err_t digit_inference_init(void) {
  if (s_interpreter != nullptr) {
    return ESP_OK;
  }

  s_model = tflite::GetModel(g_mnist_tiny_int8_model);
  if (s_model == nullptr) {
    ESP_LOGE(kTag, "Failed to parse model data");
    return ESP_FAIL;
  }

  if (s_model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "Model schema mismatch. model=%d expected=%d",
             s_model->version(), TFLITE_SCHEMA_VERSION);
    return ESP_FAIL;
  }

  if (!setup_resolver_once()) {
    return ESP_FAIL;
  }

  const size_t arena_size =
      std::max<size_t>(16384, CONFIG_DIGIT_INFERENCE_TENSOR_ARENA_SIZE);
  if (!allocate_tensor_arena(arena_size)) {
    ESP_LOGE(kTag, "Failed to allocate tensor arena (%u bytes)",
             (unsigned)arena_size);
    return ESP_ERR_NO_MEM;
  }

  s_interpreter = new (s_interpreter_storage) tflite::MicroInterpreter(
      s_model, s_resolver, s_tensor_arena, s_tensor_arena_size);

  if (s_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag,
             "AllocateTensors failed, try increasing "
             "DIGIT_INFERENCE_TENSOR_ARENA_SIZE");
    // Avoid calling MicroInterpreter destructor after a failed allocation.
    // Some TFLM versions may leave partially initialized graph state.
    s_interpreter = nullptr;
    s_input = nullptr;
    s_output = nullptr;
    free_tensor_arena();
    return ESP_ERR_NO_MEM;
  }

  s_input = s_interpreter->input(0);
  s_output = s_interpreter->output(0);

  if (s_input == nullptr || s_output == nullptr) {
    ESP_LOGE(kTag, "Interpreter tensors are not available");
    digit_inference_deinit();
    return ESP_FAIL;
  }

  ESP_LOGI(kTag,
           "TFLM initialized. arena_total=%u arena_used=%u input_type=%d "
           "output_type=%d",
           (unsigned)s_tensor_arena_size,
           (unsigned)s_interpreter->arena_used_bytes(), (int)s_input->type,
           (int)s_output->type);

  return ESP_OK;
}

extern "C" void digit_inference_deinit(void) {
  if (s_interpreter != nullptr) {
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
  }

  s_input = nullptr;
  s_output = nullptr;
  free_tensor_arena();
}

extern "C" esp_err_t digit_inference_run_u8(
    const uint8_t input_28x28[DIGIT_INFERENCE_INPUT_SIZE],
    digit_inference_result_t* result) {
  if (input_28x28 == nullptr || result == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  esp_err_t err = digit_inference_init();
  if (err != ESP_OK) {
    return err;
  }

  err = quantize_or_copy_input(input_28x28);
  if (err != ESP_OK) {
    return err;
  }

  if (s_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "TFLM invoke failed");
    return ESP_FAIL;
  }

  float probs[DIGIT_INFERENCE_NUM_CLASSES];
  err = read_output_probabilities(probs);
  if (err != ESP_OK) {
    return err;
  }

  int best_digit = 0;
  float best_score = probs[0];
  for (int i = 1; i < DIGIT_INFERENCE_NUM_CLASSES; ++i) {
    if (probs[i] > best_score) {
      best_score = probs[i];
      best_digit = i;
    }
  }

  result->predicted_digit = best_digit;
  result->confidence = best_score;
  memcpy(result->probabilities, probs, sizeof(probs));

  return ESP_OK;
}

extern "C" esp_err_t digit_inference_run_from_gray_u8(
    const uint8_t* input_gray, int width, int height,
    const digit_preprocess_params_t* params, digit_inference_result_t* result,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]) {
  if (input_gray == nullptr || result == nullptr || output_28x28 == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  esp_err_t err =
      digit_preprocess_u8(input_gray, width, height, params, output_28x28);
  if (err != ESP_OK) {
    return err;
  }

  return digit_inference_run_u8(output_28x28, result);
}
