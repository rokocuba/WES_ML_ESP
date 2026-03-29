#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#include "esp_err.h"
#include "esp_log.h"

#include "driver/uart.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "digit_inference.h"

static const char *TAG = "main";

/* UART protocol on USB serial (UART0):
 *  Host -> Device:
 *    INFER_JPEG\n
 *    <jpeg_size_bytes>\n
 *    <jpeg binary payload>
 *  Device -> Host:
 *    RESULT <digit> <confidence>\n
 *    ERR <reason>\n
 */
#define UART_PORT            UART_NUM_0
#define UART_BAUD_RATE       115200
#define UART_RX_BUF_SIZE     1024
#define UART_TX_BUF_SIZE     1024
#define UART_READ_TIMEOUT_MS 100

#define MAX_IMAGE_BYTES    (64 * 1024)
#define IMAGE_RX_TIMEOUT_MS 4000

static const char *REQUEST_COMMAND = "INFER_JPEG";
#define CMD_LINE_MAX 64

static bool uart_write_all(const void *data, size_t len)
{
    const uint8_t *ptr = (const uint8_t *)data;
    size_t sent = 0;

    while (sent < len) {
        const int written = uart_write_bytes(UART_PORT, (const char *)(ptr + sent), len - sent);
        if (written <= 0) {
            return false;
        }
        sent += (size_t)written;
    }

    return true;
}

static bool uart_write_line(const char *line)
{
    if (!line) {
        return false;
    }
    return uart_write_all(line, strlen(line));
}

static bool init_uart(void)
{
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
#if ESP_IDF_VERSION_MAJOR >= 5
        .source_clk = UART_SCLK_DEFAULT,
#endif
    };

    if (!uart_is_driver_installed(UART_PORT)) {
        esp_err_t install_err = uart_driver_install(UART_PORT, UART_RX_BUF_SIZE, UART_TX_BUF_SIZE, 0, NULL, 0);
        if (install_err != ESP_OK) {
            ESP_LOGE(TAG, "UART driver install failed: %s", esp_err_to_name(install_err));
            return false;
        }
    }

    esp_err_t err = uart_param_config(UART_PORT, &uart_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "UART param config failed: %s", esp_err_to_name(err));
        return false;
    }

    return true;
}

static bool is_infer_command(const char *line)
{
    return (strcmp(line, REQUEST_COMMAND) == 0) ||
           (strcmp(line, "RUN_INFER") == 0) ||
           (strcmp(line, "SEND_PIC") == 0);
}

static bool parse_size_line(const char *line, size_t *out_size)
{
    if (!line || !out_size || line[0] == '\0') {
        return false;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(line, &end, 10);
    if (end == line || *end != '\0') {
        return false;
    }
    if (parsed == 0 || parsed > MAX_IMAGE_BYTES) {
        return false;
    }

    *out_size = (size_t)parsed;
    return true;
}

static void run_jpeg_inference_and_respond(const uint8_t *jpeg_buf, size_t jpeg_len)
{
    const digit_preprocess_params_t *params = digit_preprocess_default_params();
    uint8_t preprocessed[DIGIT_INFERENCE_INPUT_SIZE] = {0};
    digit_inference_result_t result = {0};

    const esp_err_t ret = digit_inference_run_from_jpeg_u8(
        jpeg_buf,
        jpeg_len,
        params,
        &result,
        preprocessed
    );
    if (ret != ESP_OK) {
        char err_line[64];
        const int err_len = snprintf(err_line, sizeof(err_line), "ERR INFER %s\n", esp_err_to_name(ret));
        if (err_len > 0) {
            (void)uart_write_all(err_line, (size_t)err_len);
        }
        uart_wait_tx_done(UART_PORT, pdMS_TO_TICKS(1000));
        return;
    }

    char response[64];
    const int response_len = snprintf(
        response,
        sizeof(response),
        "RESULT %d %.3f\n",
        result.predicted_digit,
        result.confidence
    );
    if (response_len > 0) {
        (void)uart_write_all(response, (size_t)response_len);
    }
    uart_wait_tx_done(UART_PORT, pdMS_TO_TICKS(1000));
}

static void uart_inference_task(void *arg)
{
    (void)arg;

    uint8_t *rx_data = (uint8_t *)malloc(UART_RX_BUF_SIZE);
    if (!rx_data) {
        ESP_LOGE(TAG, "Failed to allocate UART RX buffer");
        vTaskDelete(NULL);
        return;
    }

    typedef enum {
        RX_WAIT_COMMAND = 0,
        RX_WAIT_SIZE,
        RX_WAIT_PAYLOAD,
    } rx_state_t;

    rx_state_t state = RX_WAIT_COMMAND;
    char line_buf[CMD_LINE_MAX];
    int line_len = 0;
    uint8_t *image_buf = NULL;
    size_t image_size = 0;
    size_t image_received = 0;
    uint32_t no_data_ms = 0;

    while (1) {
        const int rx_bytes = uart_read_bytes(UART_PORT, rx_data, UART_RX_BUF_SIZE, pdMS_TO_TICKS(UART_READ_TIMEOUT_MS));
        if (rx_bytes <= 0) {
            if (state == RX_WAIT_PAYLOAD) {
                no_data_ms += UART_READ_TIMEOUT_MS;
                if (no_data_ms >= IMAGE_RX_TIMEOUT_MS) {
                    (void)uart_write_line("ERR TIMEOUT\n");
                    uart_wait_tx_done(UART_PORT, pdMS_TO_TICKS(1000));
                    free(image_buf);
                    image_buf = NULL;
                    image_size = 0;
                    image_received = 0;
                    line_len = 0;
                    state = RX_WAIT_COMMAND;
                    uart_flush_input(UART_PORT);
                }
            }
            continue;
        }
        no_data_ms = 0;

        int offset = 0;
        while (offset < rx_bytes) {
            if (state == RX_WAIT_PAYLOAD) {
                const size_t remaining = image_size - image_received;
                const size_t available = (size_t)(rx_bytes - offset);
                const size_t copy_len = (available < remaining) ? available : remaining;

                memcpy(image_buf + image_received, rx_data + offset, copy_len);
                image_received += copy_len;
                offset += (int)copy_len;

                if (image_received == image_size) {
                    run_jpeg_inference_and_respond(image_buf, image_size);
                    free(image_buf);
                    image_buf = NULL;
                    image_size = 0;
                    image_received = 0;
                    line_len = 0;
                    state = RX_WAIT_COMMAND;
                }
                continue;
            }

            const char ch = (char)rx_data[offset++];

            if (ch == '\r') {
                continue;
            }

            if (ch == '\n') {
                line_buf[line_len] = '\0';

                if (state == RX_WAIT_COMMAND) {
                    if (is_infer_command(line_buf)) {
                        state = RX_WAIT_SIZE;
                        (void)uart_write_line("READY_SIZE\n");
                    } else if (line_buf[0] != '\0') {
                        (void)uart_write_line("ERR CMD\n");
                    }
                } else if (state == RX_WAIT_SIZE) {
                    size_t parsed_size = 0;
                    if (!parse_size_line(line_buf, &parsed_size)) {
                        (void)uart_write_line("ERR SIZE\n");
                        state = RX_WAIT_COMMAND;
                    } else {
                        image_buf = (uint8_t *)malloc(parsed_size);
                        if (!image_buf) {
                            (void)uart_write_line("ERR OOM\n");
                            state = RX_WAIT_COMMAND;
                        } else {
                            image_size = parsed_size;
                            image_received = 0;
                            state = RX_WAIT_PAYLOAD;
                            (void)uart_write_line("READY_DATA\n");
                        }
                    }
                }
                line_len = 0;
                continue;
            }

            if (line_len < (CMD_LINE_MAX - 1)) {
                line_buf[line_len++] = ch;
            } else {
                line_len = 0;
                (void)uart_write_line("ERR LINE\n");
                state = RX_WAIT_COMMAND;
                free(image_buf);
                image_buf = NULL;
                image_size = 0;
                image_received = 0;
            }
        }

        uart_wait_tx_done(UART_PORT, pdMS_TO_TICKS(100));
    }

    free(image_buf);
    free(rx_data);
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Booted. Waiting for UART image inference requests.");

    if (!init_uart()) {
        ESP_LOGE(TAG, "UART init failed");
        return;
    }

    esp_err_t ret = digit_inference_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "digit_inference_init failed: %s", esp_err_to_name(ret));
        return;
    }

    /* Prevent ESP logs from corrupting serial protocol payload/response parsing. */
    esp_log_level_set("*", ESP_LOG_NONE);

    xTaskCreate(uart_inference_task, "uart_inference_task", 6144, NULL, 5, NULL);
}