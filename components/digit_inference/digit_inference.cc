#include "./digit_inference.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "jpeg_decoder.h"
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

uint8_t* s_jpeg_decode_buffer = nullptr;
size_t s_jpeg_decode_buffer_size = 0;

struct perf_snapshot_t {
  size_t free_8bit;
  size_t largest_8bit;
  size_t minimum_8bit;
  size_t free_internal;
  size_t largest_internal;
  size_t minimum_internal;
  size_t free_spiram;
  size_t largest_spiram;
  size_t minimum_spiram;
  UBaseType_t stack_hwm_words;
};

void capture_perf_snapshot(perf_snapshot_t* snap) {
  if (snap == nullptr) {
    return;
  }

  snap->free_8bit = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  snap->largest_8bit = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  snap->minimum_8bit = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

  snap->free_internal =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  snap->largest_internal =
      heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  snap->minimum_internal =
      heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  snap->free_spiram =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  snap->largest_spiram =
      heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  snap->minimum_spiram =
      heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  snap->stack_hwm_words = uxTaskGetStackHighWaterMark(nullptr);
}

void log_pipeline_perf(const char* path_name, size_t input_bytes,
                       int input_width, int input_height, int64_t decode_us,
                       int64_t preprocess_us, int64_t inference_us,
                       const perf_snapshot_t& before,
                       const perf_snapshot_t& after) {
#if CONFIG_DIGIT_INFERENCE_PERF_LOGS
  const int64_t total_us = decode_us + preprocess_us + inference_us;
  const int32_t delta_internal_free =
      static_cast<int32_t>(after.free_internal) -
      static_cast<int32_t>(before.free_internal);
  const int32_t delta_spiram_free = static_cast<int32_t>(after.free_spiram) -
                                    static_cast<int32_t>(before.free_spiram);
  const int32_t delta_8bit_free = static_cast<int32_t>(after.free_8bit) -
                                  static_cast<int32_t>(before.free_8bit);

  ESP_LOGI(kTag,
           "Perf path=%s in=%dx%d bytes=%u ms{decode=%.2f pre=%.2f infer=%.2f "
           "total=%.2f}",
           path_name, input_width, input_height,
           static_cast<unsigned>(input_bytes),
           static_cast<double>(decode_us) / 1000.0,
           static_cast<double>(preprocess_us) / 1000.0,
           static_cast<double>(inference_us) / 1000.0,
           static_cast<double>(total_us) / 1000.0);

  ESP_LOGI(kTag,
           "Mem path=%s int{free=%u largest=%u min=%u d=%ld} psram{free=%u "
           "largest=%u min=%u d=%ld} 8bit{free=%u largest=%u min=%u d=%ld} "
           "stack_hwm_words=%u jpeg_buf{decode=%u extra_gray=%u}",
           path_name, static_cast<unsigned>(after.free_internal),
           static_cast<unsigned>(after.largest_internal),
           static_cast<unsigned>(after.minimum_internal),
           static_cast<long>(delta_internal_free),
           static_cast<unsigned>(after.free_spiram),
           static_cast<unsigned>(after.largest_spiram),
           static_cast<unsigned>(after.minimum_spiram),
           static_cast<long>(delta_spiram_free),
           static_cast<unsigned>(after.free_8bit),
           static_cast<unsigned>(after.largest_8bit),
           static_cast<unsigned>(after.minimum_8bit),
           static_cast<long>(delta_8bit_free),
           static_cast<unsigned>(after.stack_hwm_words),
           static_cast<unsigned>(s_jpeg_decode_buffer_size),
           static_cast<unsigned>(0));
#else
  (void)path_name;
  (void)input_bytes;
  (void)input_width;
  (void)input_height;
  (void)decode_us;
  (void)preprocess_us;
  (void)inference_us;
  (void)before;
  (void)after;
#endif
}

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

void free_jpeg_buffers() {
  if (s_jpeg_decode_buffer != nullptr) {
    heap_caps_free(s_jpeg_decode_buffer);
    s_jpeg_decode_buffer = nullptr;
    s_jpeg_decode_buffer_size = 0;
  }
}

bool ensure_jpeg_decode_buffer(size_t needed_size) {
  if (needed_size <= s_jpeg_decode_buffer_size &&
      s_jpeg_decode_buffer != nullptr) {
    return true;
  }

  if (s_jpeg_decode_buffer != nullptr) {
    heap_caps_free(s_jpeg_decode_buffer);
    s_jpeg_decode_buffer = nullptr;
    s_jpeg_decode_buffer_size = 0;
  }

  s_jpeg_decode_buffer =
      static_cast<uint8_t*>(heap_caps_malloc(needed_size, MALLOC_CAP_8BIT));
  if (s_jpeg_decode_buffer == nullptr) {
    ESP_LOGE(kTag, "Failed to allocate JPEG decode buffer (%u bytes)",
             static_cast<unsigned>(needed_size));
    s_jpeg_decode_buffer_size = 0;
    return false;
  }

  s_jpeg_decode_buffer_size = needed_size;
  return true;
}

esp_err_t decode_jpeg_to_gray(const uint8_t* input_jpeg, size_t jpeg_size,
                              const uint8_t** out_gray, int* out_w,
                              int* out_h) {
  if (input_jpeg == nullptr || out_gray == nullptr || out_w == nullptr ||
      out_h == nullptr || jpeg_size == 0) {
    return ESP_ERR_INVALID_ARG;
  }

  if (jpeg_size > std::numeric_limits<uint32_t>::max()) {
    return ESP_ERR_INVALID_ARG;
  }

  esp_jpeg_image_cfg_t decode_cfg = {};
  decode_cfg.indata = const_cast<uint8_t*>(input_jpeg);
  decode_cfg.indata_size = static_cast<uint32_t>(jpeg_size);
  decode_cfg.out_format = JPEG_IMAGE_FORMAT_RGB565;
  decode_cfg.out_scale = JPEG_IMAGE_SCALE_0;
  decode_cfg.flags.swap_color_bytes = 0;

  esp_jpeg_image_output_t info = {};
  esp_err_t err = esp_jpeg_get_image_info(&decode_cfg, &info);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_jpeg_get_image_info failed: %s", esp_err_to_name(err));
    return err;
  }

  const int width = static_cast<int>(info.width);
  const int height = static_cast<int>(info.height);
  if (width <= 0 || height <= 0) {
    return ESP_ERR_INVALID_SIZE;
  }

  const size_t required_decode_size = info.output_len;

  if (!ensure_jpeg_decode_buffer(required_decode_size)) {
    return ESP_ERR_NO_MEM;
  }

  decode_cfg.outbuf = s_jpeg_decode_buffer;
  decode_cfg.outbuf_size = static_cast<uint32_t>(s_jpeg_decode_buffer_size);

  esp_jpeg_image_output_t outimg = {};
  err = esp_jpeg_decode(&decode_cfg, &outimg);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_jpeg_decode failed: %s", esp_err_to_name(err));
    return err;
  }

  const size_t pixel_count =
      static_cast<size_t>(outimg.width) * static_cast<size_t>(outimg.height);
  const size_t required_rgb565 = pixel_count * 2;
  if (outimg.output_len < required_rgb565 || outimg.width == 0 ||
      outimg.height == 0) {
    ESP_LOGE(kTag, "Unexpected JPEG decode output: len=%u width=%u height=%u",
             static_cast<unsigned>(outimg.output_len),
             static_cast<unsigned>(outimg.width),
             static_cast<unsigned>(outimg.height));
    return ESP_ERR_INVALID_SIZE;
  }

  // Convert in-place from RGB565(2B/px) to grayscale(1B/px) to avoid an
  // additional full-frame allocation.
  for (size_t i = 0; i < pixel_count; ++i) {
    const uint16_t pixel565 =
        static_cast<uint16_t>(s_jpeg_decode_buffer[i * 2]) |
        (static_cast<uint16_t>(s_jpeg_decode_buffer[i * 2 + 1]) << 8);
    const uint8_t r5 = static_cast<uint8_t>((pixel565 >> 11) & 0x1F);
    const uint8_t g6 = static_cast<uint8_t>((pixel565 >> 5) & 0x3F);
    const uint8_t b5 = static_cast<uint8_t>(pixel565 & 0x1F);

    const uint8_t r8 = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
    const uint8_t g8 = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
    const uint8_t b8 = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
    const uint8_t gray =
        static_cast<uint8_t>(((77u * static_cast<uint32_t>(r8)) +
                              (150u * static_cast<uint32_t>(g8)) +
                              (29u * static_cast<uint32_t>(b8))) >>
                             8);

    s_jpeg_decode_buffer[i] = gray;
  }

  *out_gray = s_jpeg_decode_buffer;
  *out_w = static_cast<int>(outimg.width);
  *out_h = static_cast<int>(outimg.height);
  return ESP_OK;
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

uint32_t fnv1a32(const uint8_t* data, size_t len) {
  if (data == nullptr || len == 0) {
    return 0;
  }

  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= 16777619u;
  }

  return hash;
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

esp_err_t run_inference_core(
    const uint8_t input_28x28[DIGIT_INFERENCE_INPUT_SIZE],
    digit_inference_result_t* result) {
  esp_err_t err = quantize_or_copy_input(input_28x28);
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

}  // namespace

extern "C" esp_err_t digit_inference_init(void) {
  if (s_interpreter != nullptr) {
    return ESP_OK;
  }

  if (g_mnist_tiny_int8_model_len == 0) {
    ESP_LOGE(kTag, "Model buffer is empty");
    return ESP_FAIL;
  }

  const uint32_t model_hash =
      fnv1a32(g_mnist_tiny_int8_model, g_mnist_tiny_int8_model_len);
  if (g_mnist_tiny_int8_model_len >= 4) {
    ESP_LOGI(kTag,
             "Model fingerprint len=%u fnv1a32=0x%08" PRIx32
             " head=%02x%02x%02x%02x",
             static_cast<unsigned>(g_mnist_tiny_int8_model_len), model_hash,
             static_cast<unsigned>(g_mnist_tiny_int8_model[0]),
             static_cast<unsigned>(g_mnist_tiny_int8_model[1]),
             static_cast<unsigned>(g_mnist_tiny_int8_model[2]),
             static_cast<unsigned>(g_mnist_tiny_int8_model[3]));
  } else {
    ESP_LOGI(kTag, "Model fingerprint len=%u fnv1a32=0x%08" PRIx32,
             static_cast<unsigned>(g_mnist_tiny_int8_model_len), model_hash);
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

  free_jpeg_buffers();
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

  perf_snapshot_t before = {};
  perf_snapshot_t after = {};
  capture_perf_snapshot(&before);

  const int64_t t_infer_begin = esp_timer_get_time();
  err = run_inference_core(input_28x28, result);
  const int64_t t_infer_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  capture_perf_snapshot(&after);
  log_pipeline_perf("u8", DIGIT_INFERENCE_INPUT_SIZE,
                    DIGIT_INFERENCE_INPUT_SIDE, DIGIT_INFERENCE_INPUT_SIDE, 0,
                    0, t_infer_end - t_infer_begin, before, after);

  return ESP_OK;
}

extern "C" esp_err_t digit_inference_run_from_gray_u8(
    const uint8_t* input_gray, int width, int height,
    const digit_preprocess_params_t* params, digit_inference_result_t* result,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]) {
  if (input_gray == nullptr || result == nullptr || output_28x28 == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  esp_err_t err = digit_inference_init();
  if (err != ESP_OK) {
    return err;
  }

  perf_snapshot_t before = {};
  perf_snapshot_t after = {};
  capture_perf_snapshot(&before);

  const int64_t t_pre_begin = esp_timer_get_time();
  err = digit_preprocess_u8(input_gray, width, height, params, output_28x28);
  const int64_t t_pre_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  const int64_t t_infer_begin = esp_timer_get_time();
  err = run_inference_core(output_28x28, result);
  const int64_t t_infer_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  capture_perf_snapshot(&after);
  log_pipeline_perf("gray",
                    static_cast<size_t>(width) * static_cast<size_t>(height),
                    width, height, 0, t_pre_end - t_pre_begin,
                    t_infer_end - t_infer_begin, before, after);

  return ESP_OK;
}

extern "C" esp_err_t digit_inference_run_from_jpeg_u8(
    const uint8_t* input_jpeg, size_t jpeg_size,
    const digit_preprocess_params_t* params, digit_inference_result_t* result,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]) {
  if (input_jpeg == nullptr || jpeg_size == 0 || result == nullptr ||
      output_28x28 == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  esp_err_t err = digit_inference_init();
  if (err != ESP_OK) {
    return err;
  }

  perf_snapshot_t before = {};
  perf_snapshot_t after = {};
  capture_perf_snapshot(&before);

  const int64_t t_decode_begin = esp_timer_get_time();

  const uint8_t* gray = nullptr;
  int width = 0;
  int height = 0;
  err = decode_jpeg_to_gray(input_jpeg, jpeg_size, &gray, &width, &height);
  const int64_t t_decode_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  const int64_t t_pre_begin = esp_timer_get_time();
  err = digit_preprocess_u8(gray, width, height, params, output_28x28);
  const int64_t t_pre_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  const int64_t t_infer_begin = esp_timer_get_time();
  err = run_inference_core(output_28x28, result);
  const int64_t t_infer_end = esp_timer_get_time();
  if (err != ESP_OK) {
    return err;
  }

  capture_perf_snapshot(&after);
  log_pipeline_perf("jpeg", jpeg_size, width, height,
                    t_decode_end - t_decode_begin, t_pre_end - t_pre_begin,
                    t_infer_end - t_infer_begin, before, after);

  return ESP_OK;
}
