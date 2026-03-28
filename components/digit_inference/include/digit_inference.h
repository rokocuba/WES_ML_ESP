#ifndef DIGIT_INFERENCE_H_
#define DIGIT_INFERENCE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DIGIT_INFERENCE_INPUT_SIDE (28)
#define DIGIT_INFERENCE_INPUT_SIZE \
  (DIGIT_INFERENCE_INPUT_SIDE * DIGIT_INFERENCE_INPUT_SIDE)
#define DIGIT_INFERENCE_NUM_CLASSES (10)

typedef struct {
  int process_size;
  int window_size;
  float threshold_t;
  float relax_delta;
  bool remove_border;
  int target_size;
  int box_size;
  int margin;
  float fit_foreground_threshold_ratio;
} digit_preprocess_params_t;

typedef struct {
  int predicted_digit;
  float confidence;
  float probabilities[DIGIT_INFERENCE_NUM_CLASSES];
} digit_inference_result_t;

const digit_preprocess_params_t* digit_preprocess_default_params(void);

esp_err_t digit_preprocess_u8(const uint8_t* input_gray, int width, int height,
                              const digit_preprocess_params_t* params,
                              uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]);

esp_err_t digit_inference_init(void);
void digit_inference_deinit(void);

esp_err_t digit_inference_run_u8(
    const uint8_t input_28x28[DIGIT_INFERENCE_INPUT_SIZE],
    digit_inference_result_t* result);

esp_err_t digit_inference_run_from_gray_u8(
    const uint8_t* input_gray, int width, int height,
    const digit_preprocess_params_t* params, digit_inference_result_t* result,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]);

esp_err_t digit_inference_run_from_jpeg_u8(
    const uint8_t* input_jpeg, size_t jpeg_size,
    const digit_preprocess_params_t* params, digit_inference_result_t* result,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]);

#ifdef __cplusplus
}
#endif

#endif /* DIGIT_INFERENCE_H_ */
