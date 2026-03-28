#include "./digit_inference.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define DIGIT_DIM 28
#define DIGIT_PIXELS (DIGIT_DIM * DIGIT_DIM)
#define INTEGRAL_SIDE (DIGIT_DIM + 1)

typedef struct {
    uint8_t resized[DIGIT_PIXELS];
    uint8_t inverted[DIGIT_PIXELS];
    uint8_t fitted[DIGIT_PIXELS];
    uint8_t strong[DIGIT_PIXELS];
    uint8_t weak[DIGIT_PIXELS];
    uint8_t linked[DIGIT_PIXELS];
    uint8_t border_seed[DIGIT_PIXELS];
    uint8_t border_connected[DIGIT_PIXELS];
    uint8_t temp_mask[DIGIT_PIXELS];
    int queue[DIGIT_PIXELS];
    uint32_t integral[INTEGRAL_SIDE * INTEGRAL_SIDE];
} preprocess_workspace_t;

static preprocess_workspace_t s_ws;

static const digit_preprocess_params_t k_default_params = {
    .process_size = DIGIT_DIM,
    .window_size = 21,
    .threshold_t = 0.12f,
    .relax_delta = 0.10f,
    .remove_border = true,
    .target_size = DIGIT_DIM,
    .box_size = 20,
    .margin = 1,
    .fit_foreground_threshold_ratio = 0.01f,
};

const digit_preprocess_params_t *digit_preprocess_default_params(void)
{
    return &k_default_params;
}

static inline int idx_2d(int x, int y)
{
    return y * DIGIT_DIM + x;
}

static inline int clampi(int value, int lo, int hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static inline float clampf(float value, float lo, float hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static void resize_bilinear_u8(
    const uint8_t *src,
    int src_w,
    int src_h,
    uint8_t *dst,
    int dst_w,
    int dst_h
)
{
    for (int y = 0; y < dst_h; ++y) {
        const float gy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
        int y0 = (int)floorf(gy);
        int y1 = y0 + 1;
        const float wy = gy - (float)y0;

        y0 = clampi(y0, 0, src_h - 1);
        y1 = clampi(y1, 0, src_h - 1);

        for (int x = 0; x < dst_w; ++x) {
            const float gx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
            int x0 = (int)floorf(gx);
            int x1 = x0 + 1;
            const float wx = gx - (float)x0;

            x0 = clampi(x0, 0, src_w - 1);
            x1 = clampi(x1, 0, src_w - 1);

            const float p00 = (float)src[y0 * src_w + x0];
            const float p01 = (float)src[y0 * src_w + x1];
            const float p10 = (float)src[y1 * src_w + x0];
            const float p11 = (float)src[y1 * src_w + x1];

            const float top = p00 + (p01 - p00) * wx;
            const float bottom = p10 + (p11 - p10) * wx;
            const int out = (int)lroundf(top + (bottom - top) * wy);
            dst[y * dst_w + x] = (uint8_t)clampi(out, 0, 255);
        }
    }
}

static void build_integral_u8(const uint8_t *gray, uint32_t *integral)
{
    memset(integral, 0, sizeof(uint32_t) * INTEGRAL_SIDE * INTEGRAL_SIDE);

    for (int y = 0; y < DIGIT_DIM; ++y) {
        uint32_t row_sum = 0;
        for (int x = 0; x < DIGIT_DIM; ++x) {
            row_sum += (uint32_t)gray[idx_2d(x, y)];
            integral[(y + 1) * INTEGRAL_SIDE + (x + 1)] =
                integral[y * INTEGRAL_SIDE + (x + 1)] + row_sum;
        }
    }
}

static void bradley_roth_mask_28(
    const uint8_t *gray,
    int window_size,
    float t_value,
    uint8_t *mask,
    uint32_t *integral
)
{
    build_integral_u8(gray, integral);

    int local_window = window_size;
    if (local_window < 3) {
        local_window = 3;
    }
    if ((local_window % 2) == 0) {
        local_window += 1;
    }

    const int radius = local_window / 2;
    const float t = clampf(t_value, 0.0f, 0.99f);

    for (int y = 0; y < DIGIT_DIM; ++y) {
        const int y0 = clampi(y - radius, 0, DIGIT_DIM - 1);
        const int y1 = clampi(y + radius, 0, DIGIT_DIM - 1);

        for (int x = 0; x < DIGIT_DIM; ++x) {
            const int x0 = clampi(x - radius, 0, DIGIT_DIM - 1);
            const int x1 = clampi(x + radius, 0, DIGIT_DIM - 1);

            const int i_y0 = y0;
            const int i_y1 = y1 + 1;
            const int i_x0 = x0;
            const int i_x1 = x1 + 1;

            const uint32_t local_sum =
                integral[i_y1 * INTEGRAL_SIDE + i_x1]
                - integral[i_y0 * INTEGRAL_SIDE + i_x1]
                - integral[i_y1 * INTEGRAL_SIDE + i_x0]
                + integral[i_y0 * INTEGRAL_SIDE + i_x0];

            const int area = (y1 - y0 + 1) * (x1 - x0 + 1);
            const float lhs = (float)gray[idx_2d(x, y)] * (float)area;
            const float rhs = (float)local_sum * (1.0f - t);
            mask[idx_2d(x, y)] = (lhs <= rhs) ? 1 : 0;
        }
    }
}

static bool mask_has_foreground(const uint8_t *mask)
{
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        if (mask[i]) {
            return true;
        }
    }
    return false;
}

static void flood_fill_from_seed(
    const uint8_t *seed,
    const uint8_t *allowed,
    uint8_t *out,
    int *queue
)
{
    memset(out, 0, DIGIT_PIXELS);

    int head = 0;
    int tail = 0;

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        if (seed[i] && allowed[i]) {
            out[i] = 1;
            queue[tail++] = i;
        }
    }

    while (head < tail) {
        const int cur = queue[head++];
        const int x = cur % DIGIT_DIM;
        const int y = cur / DIGIT_DIM;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx < 0 || nx >= DIGIT_DIM || ny < 0 || ny >= DIGIT_DIM) {
                    continue;
                }
                const int nidx = idx_2d(nx, ny);
                if (allowed[nidx] && !out[nidx]) {
                    out[nidx] = 1;
                    queue[tail++] = nidx;
                }
            }
        }
    }
}

static void remove_border_connected_inplace(
    uint8_t *mask,
    uint8_t *border_seed,
    uint8_t *border_connected,
    int *queue
)
{
    memset(border_seed, 0, DIGIT_PIXELS);

    for (int x = 0; x < DIGIT_DIM; ++x) {
        border_seed[idx_2d(x, 0)] = mask[idx_2d(x, 0)];
        border_seed[idx_2d(x, DIGIT_DIM - 1)] = mask[idx_2d(x, DIGIT_DIM - 1)];
    }

    for (int y = 0; y < DIGIT_DIM; ++y) {
        border_seed[idx_2d(0, y)] |= mask[idx_2d(0, y)];
        border_seed[idx_2d(DIGIT_DIM - 1, y)] |= mask[idx_2d(DIGIT_DIM - 1, y)];
    }

    flood_fill_from_seed(border_seed, mask, border_connected, queue);

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        if (border_connected[i]) {
            mask[i] = 0;
        }
    }
}

static void fit_digit_to_canvas_28(
    const uint8_t *input,
    int box_size,
    int margin,
    float foreground_threshold_ratio,
    uint8_t *output
)
{
    uint8_t crop[DIGIT_PIXELS];
    uint8_t resized[DIGIT_PIXELS];

    const int threshold =
        clampi((int)lroundf(clampf(foreground_threshold_ratio, 0.0f, 1.0f) * 255.0f), 0, 255);

    int min_x = DIGIT_DIM;
    int min_y = DIGIT_DIM;
    int max_x = -1;
    int max_y = -1;

    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            if (input[idx_2d(x, y)] > threshold) {
                if (x < min_x) {
                    min_x = x;
                }
                if (x > max_x) {
                    max_x = x;
                }
                if (y < min_y) {
                    min_y = y;
                }
                if (y > max_y) {
                    max_y = y;
                }
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        memcpy(output, input, DIGIT_PIXELS);
        return;
    }

    const int crop_w = max_x - min_x + 1;
    const int crop_h = max_y - min_y + 1;

    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            crop[y * crop_w + x] = input[idx_2d(min_x + x, min_y + y)];
        }
    }

    int local_margin = margin;
    if (local_margin < 0) {
        local_margin = 0;
    }
    if (local_margin >= (DIGIT_DIM / 2)) {
        local_margin = (DIGIT_DIM / 2) - 1;
    }

    int max_box = DIGIT_DIM - (2 * local_margin);
    if (max_box < 1) {
        max_box = 1;
    }

    const int target_long = clampi(box_size, 1, max_box);
    const int long_side = (crop_h > crop_w) ? crop_h : crop_w;
    const float scale = (float)target_long / (float)((long_side > 0) ? long_side : 1);

    const int new_h = clampi((int)lroundf((float)crop_h * scale), 1, DIGIT_DIM);
    const int new_w = clampi((int)lroundf((float)crop_w * scale), 1, DIGIT_DIM);

    resize_bilinear_u8(crop, crop_w, crop_h, resized, new_w, new_h);

    memset(output, 0, DIGIT_PIXELS);

    const int center_y = (DIGIT_DIM - new_h) / 2;
    const int center_x = (DIGIT_DIM - new_w) / 2;

    int y_min = local_margin;
    int x_min = local_margin;
    int y_max = DIGIT_DIM - new_h - local_margin;
    int x_max = DIGIT_DIM - new_w - local_margin;

    if (y_max < y_min) {
        y_min = 0;
        y_max = DIGIT_DIM - new_h;
    }
    if (x_max < x_min) {
        x_min = 0;
        x_max = DIGIT_DIM - new_w;
    }

    const int y_start = clampi(center_y, y_min, y_max);
    const int x_start = clampi(center_x, x_min, x_max);

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            output[idx_2d(x_start + x, y_start + y)] = resized[y * new_w + x];
        }
    }
}

esp_err_t digit_preprocess_u8(
    const uint8_t *input_gray,
    int width,
    int height,
    const digit_preprocess_params_t *params,
    uint8_t output_28x28[DIGIT_INFERENCE_INPUT_SIZE]
)
{
    if (input_gray == NULL || output_28x28 == NULL || width <= 0 || height <= 0) {
        return ESP_ERR_INVALID_ARG;
    }

    const digit_preprocess_params_t *p = params;
    if (p == NULL) {
        p = digit_preprocess_default_params();
    }

    if (p->process_size != DIGIT_DIM || p->target_size != DIGIT_DIM) {
        return ESP_ERR_NOT_SUPPORTED;
    }

    int window_size = p->window_size;
    if (window_size < 3) {
        window_size = 3;
    }
    if ((window_size % 2) == 0) {
        window_size += 1;
    }

    resize_bilinear_u8(input_gray, width, height, s_ws.resized, DIGIT_DIM, DIGIT_DIM);

    bradley_roth_mask_28(
        s_ws.resized,
        window_size,
        p->threshold_t,
        s_ws.strong,
        s_ws.integral
    );

    bradley_roth_mask_28(
        s_ws.resized,
        window_size,
        p->threshold_t - p->relax_delta,
        s_ws.weak,
        s_ws.integral
    );

    flood_fill_from_seed(s_ws.strong, s_ws.weak, s_ws.linked, s_ws.queue);
    if (!mask_has_foreground(s_ws.linked)) {
        memcpy(s_ws.linked, s_ws.strong, DIGIT_PIXELS);
    }

    if (p->remove_border) {
        memcpy(s_ws.temp_mask, s_ws.linked, DIGIT_PIXELS);
        remove_border_connected_inplace(
            s_ws.temp_mask,
            s_ws.border_seed,
            s_ws.border_connected,
            s_ws.queue
        );
        if (mask_has_foreground(s_ws.temp_mask)) {
            memcpy(s_ws.linked, s_ws.temp_mask, DIGIT_PIXELS);
        }
    }

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        const uint8_t foreground_gray = s_ws.linked[i] ? s_ws.resized[i] : 255;
        s_ws.inverted[i] = (uint8_t)(255 - foreground_gray);
    }

    fit_digit_to_canvas_28(
        s_ws.inverted,
        p->box_size,
        p->margin,
        p->fit_foreground_threshold_ratio,
        s_ws.fitted
    );

    memcpy(output_28x28, s_ws.fitted, DIGIT_PIXELS);
    return ESP_OK;
}
