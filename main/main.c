#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "digit_inference.h"
#include "digit_test_images_data.h"

static const char *TAG = "main";

static void log_top3(const digit_inference_result_t *result)
{
	int best = -1;
	int second = -1;
	int third = -1;

	for (int i = 0; i < DIGIT_INFERENCE_NUM_CLASSES; ++i) {
		if (best < 0 || result->probabilities[i] > result->probabilities[best]) {
			third = second;
			second = best;
			best = i;
		} else if (second < 0 || result->probabilities[i] > result->probabilities[second]) {
			third = second;
			second = i;
		} else if (third < 0 || result->probabilities[i] > result->probabilities[third]) {
			third = i;
		}
	}

	if (best >= 0 && second >= 0 && third >= 0) {
		ESP_LOGI(
			TAG,
			"Top3 => %d:%.3f, %d:%.3f, %d:%.3f",
			best, result->probabilities[best],
			second, result->probabilities[second],
			third, result->probabilities[third]
		);
	}
}

static void log_preprocessed_frame_hex(const char *name, const uint8_t frame[DIGIT_INFERENCE_INPUT_SIZE])
{
	char row_hex[(DIGIT_INFERENCE_INPUT_SIDE * 2) + 1];

	ESP_LOGI(
		TAG,
		"Preproc28 BEGIN name=%s w=%d h=%d format=hex",
		name,
		DIGIT_INFERENCE_INPUT_SIDE,
		DIGIT_INFERENCE_INPUT_SIDE
	);

	for (int y = 0; y < DIGIT_INFERENCE_INPUT_SIDE; ++y) {
		for (int x = 0; x < DIGIT_INFERENCE_INPUT_SIDE; ++x) {
			const uint8_t px = frame[y * DIGIT_INFERENCE_INPUT_SIDE + x];
			(void)snprintf(&row_hex[x * 2], 3, "%02x", px);
		}
		row_hex[DIGIT_INFERENCE_INPUT_SIDE * 2] = '\0';
		ESP_LOGI(TAG, "Preproc28 ROW%02d %s", y, row_hex);
	}

	ESP_LOGI(TAG, "Preproc28 END name=%s", name);
}

static void run_embedded_png_once(size_t image_index, bool dump_preprocessed)
{
	const digit_preprocess_params_t *params = digit_preprocess_default_params();
	uint8_t preprocessed[DIGIT_INFERENCE_INPUT_SIZE] = {0};
	digit_inference_result_t result = {0};
	const digit_test_image_t *image = &g_digit_test_images[image_index];

	memset(&result, 0, sizeof(result));

	esp_err_t ret = digit_inference_run_from_gray_u8(
		image->data,
		image->width,
		image->height,
		params,
		&result,
		preprocessed
	);
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "EmbeddedPNG[%u]=%s inference failed: %s", (unsigned)image_index, image->name, esp_err_to_name(ret));
		return;
	}

	ESP_LOGI(
		TAG,
		"EmbeddedPNG[%u]=%s (%dx%d) -> pred=%d conf=%.3f",
		(unsigned)image_index,
		image->name,
		image->width,
		image->height,
		result.predicted_digit,
		result.confidence
	);
	log_top3(&result);

	if (dump_preprocessed) {
		log_preprocessed_frame_hex(image->name, preprocessed);
	}
}

static void png_loop_task(void *arg)
{
	uint32_t cycle = 0;
	(void)arg;

	while (1) {
		ESP_LOGI(TAG, "PNG loop cycle=%lu count=%u", (unsigned long)cycle, (unsigned)g_digit_test_images_count);

		for (size_t i = 0; i < g_digit_test_images_count; ++i) {
			const bool dump_preprocessed = (cycle == 0);
			run_embedded_png_once(i, dump_preprocessed);
		}

		++cycle;
		vTaskDelay(pdMS_TO_TICKS(2000));
	}
}

void app_main(void)
{
	ESP_LOGI(TAG, "Booted. Starting embedded PNG inference loop.");

	esp_err_t ret = digit_inference_init();
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "digit_inference_init failed: %s", esp_err_to_name(ret));
		return;
	}

	xTaskCreate(png_loop_task, "png_loop_task", 6144, NULL, 5, NULL);
}
