#include <stdio.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "digit_inference.h"

static const char *TAG = "main";

static void generate_demo_pattern(int pattern_id, uint8_t sample[DIGIT_INFERENCE_INPUT_SIZE])
{
	memset(sample, 0, DIGIT_INFERENCE_INPUT_SIZE);

	if (pattern_id == 1) {
		/* Center block */
		for (int y = 10; y < 18; ++y) {
			for (int x = 10; x < 18; ++x) {
				sample[y * DIGIT_INFERENCE_INPUT_SIDE + x] = 255;
			}
		}
		return;
	}

	if (pattern_id == 2) {
		/* Ring-like shape */
		const int cx = 14;
		const int cy = 14;
		for (int y = 0; y < DIGIT_INFERENCE_INPUT_SIDE; ++y) {
			for (int x = 0; x < DIGIT_INFERENCE_INPUT_SIDE; ++x) {
				const int dx = x - cx;
				const int dy = y - cy;
				const int d2 = dx * dx + dy * dy;
				if (d2 >= 36 && d2 <= 64) {
					sample[y * DIGIT_INFERENCE_INPUT_SIDE + x] = 255;
				}
			}
		}
		return;
	}

	if (pattern_id == 3) {
		/* Rough "5"-like stroke pattern */
		for (int x = 6; x <= 21; ++x) {
			sample[6 * DIGIT_INFERENCE_INPUT_SIDE + x] = 255;
			sample[14 * DIGIT_INFERENCE_INPUT_SIDE + x] = 255;
			sample[22 * DIGIT_INFERENCE_INPUT_SIDE + x] = 255;
		}
		for (int y = 6; y <= 14; ++y) {
			sample[y * DIGIT_INFERENCE_INPUT_SIDE + 6] = 255;
		}
		for (int y = 14; y <= 22; ++y) {
			sample[y * DIGIT_INFERENCE_INPUT_SIDE + 21] = 255;
		}
	}
}

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

static void run_digit_inference_smoke_test(void)
{
	uint8_t sample[DIGIT_INFERENCE_INPUT_SIZE] = {0};
	digit_inference_result_t result = {0};

	esp_err_t ret = digit_inference_init();
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "digit_inference_init failed: %s", esp_err_to_name(ret));
		return;
	}

	ret = digit_inference_run_u8(sample, &result);
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "digit_inference_run_u8 failed: %s", esp_err_to_name(ret));
		return;
	}

	ESP_LOGI(
		TAG,
		"Digit inference smoke test -> pred=%d, conf=%.3f",
		result.predicted_digit,
		result.confidence
	);
	log_top3(&result);
}

static void demo_task(void *arg)
{
	uint32_t tick = 0;
	uint8_t sample[DIGIT_INFERENCE_INPUT_SIZE];
	digit_inference_result_t result;

	const char *pattern_names[] = {
		"blank",
		"center_block",
		"ring",
		"five_like"
	};

	while (1) {
		ESP_LOGI(TAG, "Heartbeat tick=%lu", (unsigned long)tick++);

		for (int pattern = 0; pattern < 4; ++pattern) {
			generate_demo_pattern(pattern, sample);
			memset(&result, 0, sizeof(result));

			esp_err_t ret = digit_inference_run_u8(sample, &result);
			if (ret != ESP_OK) {
				ESP_LOGE(TAG, "Inference failed for %s: %s", pattern_names[pattern], esp_err_to_name(ret));
				continue;
			}

			ESP_LOGI(
				TAG,
				"Pattern=%s -> pred=%d conf=%.3f",
				pattern_names[pattern],
				result.predicted_digit,
				result.confidence
			);
			log_top3(&result);
		}

		vTaskDelay(pdMS_TO_TICKS(2000));
	}
}

void app_main(void)
{
	ESP_LOGI(TAG, "Booted. Starting digit inference demo task.");
	run_digit_inference_smoke_test();
	xTaskCreate(demo_task, "demo_task", 4096, NULL, 5, NULL);
}
