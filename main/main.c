#include <stdio.h>

#include "esp_err.h"
#include "esp_log.h"

#include "digit_inference.h"

static const char *TAG = "main";

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
}

void app_main(void)
{
	run_digit_inference_smoke_test();
}
