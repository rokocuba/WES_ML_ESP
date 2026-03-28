#include "./digit_inference.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DIGIT_DIM 28
#define DIGIT_PIXELS (DIGIT_DIM * DIGIT_DIM)
#define MASK_REGION_MIN 0.2f
#define LINK_MASK_MIN 0.12f

typedef struct {
    float resized[DIGIT_PIXELS];
    float mask[DIGIT_PIXELS];
    float stage1[DIGIT_PIXELS];
    float stage2[DIGIT_PIXELS];
    float stage3[DIGIT_PIXELS];
    float darkness[DIGIT_PIXELS];
    float cloud[DIGIT_PIXELS];
    float cloud_component[DIGIT_PIXELS];
    float response[DIGIT_PIXELS];
    float link_response[DIGIT_PIXELS];
    float blur_tmp[DIGIT_PIXELS];
    float sort_scratch[DIGIT_PIXELS];
    uint8_t strong[DIGIT_PIXELS];
    uint8_t weak[DIGIT_PIXELS];
    uint8_t linked[DIGIT_PIXELS];
    uint8_t morph_tmp[DIGIT_PIXELS];
    int queue[DIGIT_PIXELS];
} preprocess_workspace_t;

static preprocess_workspace_t s_ws;

static const digit_preprocess_params_t k_default_params = {
    .circle_radius = 13.0f,
    .transition_width = 3.5f,
    .top_bright_frac = 0.10f,
    .cloud_sigma = 2.2f,
    .cloud_strength = 0.95f,
    .sharpen_radius = 0.9f,
    .sharpen_amount = 0.22f,
    .threshold_k = 0.70f,
    .threshold_floor = 0.03f,
    .threshold_soft_width = 0.08f,
    .stroke_link_margin = 0.045f,
};

const digit_preprocess_params_t *digit_preprocess_default_params(void)
{
    return &k_default_params;
}

static inline int idx_2d(int x, int y)
{
    return y * DIGIT_DIM + x;
}

static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int compare_float_asc(const void *a, const void *b)
{
    const float fa = *(const float *)a;
    const float fb = *(const float *)b;
    if (fa < fb) {
        return -1;
    }
    if (fa > fb) {
        return 1;
    }
    return 0;
}

static float percentile_from_sorted(const float *sorted, int count, float percentile)
{
    if (count <= 0) {
        return 0.0f;
    }

    const float p = clampf(percentile, 0.0f, 100.0f) * 0.01f;
    const float idx = p * (float)(count - 1);
    const int lo = (int)floorf(idx);
    const int hi = (int)ceilf(idx);
    if (lo == hi) {
        return sorted[lo];
    }

    const float alpha = idx - (float)lo;
    return sorted[lo] * (1.0f - alpha) + sorted[hi] * alpha;
}

static int collect_masked_values(
    const float *values,
    const float *mask,
    float mask_min,
    float *out
)
{
    int count = 0;
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        if (mask[i] >= mask_min) {
            out[count++] = values[i];
        }
    }
    return count;
}

static float top_bright_mean(const float *image, float fraction, float *scratch)
{
    const float frac = clampf(fraction, 0.001f, 1.0f);
    const int top_n = (int)lroundf(frac * (float)DIGIT_PIXELS);
    const int n = top_n < 1 ? 1 : (top_n > DIGIT_PIXELS ? DIGIT_PIXELS : top_n);

    memcpy(scratch, image, sizeof(float) * DIGIT_PIXELS);
    qsort(scratch, DIGIT_PIXELS, sizeof(float), compare_float_asc);

    float sum = 0.0f;
    for (int i = DIGIT_PIXELS - n; i < DIGIT_PIXELS; ++i) {
        sum += scratch[i];
    }
    return sum / (float)n;
}

static float percentile_masked(
    const float *values,
    const float *mask,
    float mask_min,
    float percentile,
    float *scratch
)
{
    int count = collect_masked_values(values, mask, mask_min, scratch);
    if (count <= 0) {
        memcpy(scratch, values, sizeof(float) * DIGIT_PIXELS);
        count = DIGIT_PIXELS;
    }

    qsort(scratch, count, sizeof(float), compare_float_asc);
    return percentile_from_sorted(scratch, count, percentile);
}

static void mean_std_masked(
    const float *values,
    const float *mask,
    float mask_min,
    float *mean_out,
    float *std_out,
    float *scratch
)
{
    int count = collect_masked_values(values, mask, mask_min, scratch);
    if (count <= 0) {
        memcpy(scratch, values, sizeof(float) * DIGIT_PIXELS);
        count = DIGIT_PIXELS;
    }

    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += scratch[i];
    }
    const float mean = sum / (float)count;

    float var = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float d = scratch[i] - mean;
        var += d * d;
    }
    var /= (float)count;

    *mean_out = mean;
    *std_out = sqrtf(var);
}

static float otsu_threshold(const float *values, int count)
{
    if (count <= 0) {
        return 0.5f;
    }

    uint32_t hist[256];
    memset(hist, 0, sizeof(hist));

    for (int i = 0; i < count; ++i) {
        int q = (int)lroundf(clampf(values[i], 0.0f, 1.0f) * 255.0f);
        if (q < 0) {
            q = 0;
        }
        if (q > 255) {
            q = 255;
        }
        hist[q]++;
    }

    const float total = (float)count;
    float sum_total = 0.0f;
    for (int i = 0; i < 256; ++i) {
        sum_total += (float)i * (float)hist[i];
    }

    float sum_bg = 0.0f;
    float weight_bg = 0.0f;
    float max_between = -1.0f;
    int threshold = 127;

    for (int i = 0; i < 256; ++i) {
        weight_bg += (float)hist[i];
        if (weight_bg <= 0.0f) {
            continue;
        }

        const float weight_fg = total - weight_bg;
        if (weight_fg <= 0.0f) {
            break;
        }

        sum_bg += (float)i * (float)hist[i];
        const float mean_bg = sum_bg / weight_bg;
        const float mean_fg = (sum_total - sum_bg) / weight_fg;
        const float diff = mean_bg - mean_fg;
        const float between = weight_bg * weight_fg * diff * diff;
        if (between > max_between) {
            max_between = between;
            threshold = i;
        }
    }

    return (float)threshold / 255.0f;
}

static void resize_bilinear_to_28(const uint8_t *src, int src_w, int src_h, float *dst)
{
    for (int y = 0; y < DIGIT_DIM; ++y) {
        const float gy = ((float)y + 0.5f) * (float)src_h / (float)DIGIT_DIM - 0.5f;
        int y0 = (int)floorf(gy);
        int y1 = y0 + 1;
        const float wy = gy - (float)y0;

        if (y0 < 0) {
            y0 = 0;
        }
        if (y1 >= src_h) {
            y1 = src_h - 1;
        }

        for (int x = 0; x < DIGIT_DIM; ++x) {
            const float gx = ((float)x + 0.5f) * (float)src_w / (float)DIGIT_DIM - 0.5f;
            int x0 = (int)floorf(gx);
            int x1 = x0 + 1;
            const float wx = gx - (float)x0;

            if (x0 < 0) {
                x0 = 0;
            }
            if (x1 >= src_w) {
                x1 = src_w - 1;
            }

            const float p00 = (float)src[y0 * src_w + x0] / 255.0f;
            const float p01 = (float)src[y0 * src_w + x1] / 255.0f;
            const float p10 = (float)src[y1 * src_w + x0] / 255.0f;
            const float p11 = (float)src[y1 * src_w + x1] / 255.0f;

            const float top = p00 + (p01 - p00) * wx;
            const float bottom = p10 + (p11 - p10) * wx;
            dst[idx_2d(x, y)] = clampf(top + (bottom - top) * wy, 0.0f, 1.0f);
        }
    }
}

static void build_soft_circle_mask(float radius, float transition_width, float *mask)
{
    const float outer = radius > 1.0f ? radius : 1.0f;
    const float tw = transition_width > 0.1f ? transition_width : 0.1f;
    const float softness = clampf(tw / outer, 0.05f, 1.0f);
    const float gamma = 1.15f - 0.60f * softness;

    const float cx = ((float)DIGIT_DIM - 1.0f) * 0.5f;
    const float cy = ((float)DIGIT_DIM - 1.0f) * 0.5f;

    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            const float dx = (float)x - cx;
            const float dy = (float)y - cy;
            const float dist = sqrtf(dx * dx + dy * dy);

            const float radial = clampf(1.0f - dist / outer, 0.0f, 1.0f);
            const float smooth = radial * radial * (3.0f - 2.0f * radial);
            mask[idx_2d(x, y)] = powf(clampf(smooth, 0.0f, 1.0f), gamma);
        }
    }
}

static int make_gaussian_kernel(float sigma, float *kernel, int kernel_cap)
{
    const float sig = sigma > 0.3f ? sigma : 0.3f;
    int radius = (int)ceilf(3.0f * sig);
    if (radius < 1) {
        radius = 1;
    }
    if (radius > 6) {
        radius = 6;
    }

    int len = radius * 2 + 1;
    if (len > kernel_cap) {
        len = kernel_cap;
        radius = len / 2;
    }

    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = expf(-(float)(i * i) / (2.0f * sig * sig));
        kernel[i + radius] = v;
        sum += v;
    }

    if (sum <= 1e-9f) {
        sum = 1.0f;
    }

    for (int i = 0; i < len; ++i) {
        kernel[i] /= sum;
    }

    return len;
}

static void gaussian_blur_28(const float *src, float *dst, float sigma, float *tmp)
{
    if (sigma <= 0.0f) {
        memcpy(dst, src, sizeof(float) * DIGIT_PIXELS);
        return;
    }

    float kernel[13];
    const int len = make_gaussian_kernel(sigma, kernel, 13);
    const int radius = len / 2;

    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sx = x + k;
                if (sx < 0) {
                    sx = 0;
                }
                if (sx >= DIGIT_DIM) {
                    sx = DIGIT_DIM - 1;
                }
                acc += src[idx_2d(sx, y)] * kernel[k + radius];
            }
            tmp[idx_2d(x, y)] = acc;
        }
    }

    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sy = y + k;
                if (sy < 0) {
                    sy = 0;
                }
                if (sy >= DIGIT_DIM) {
                    sy = DIGIT_DIM - 1;
                }
                acc += tmp[idx_2d(x, sy)] * kernel[k + radius];
            }
            dst[idx_2d(x, y)] = acc;
        }
    }
}

static void binary_propagate(
    const uint8_t *strong,
    const uint8_t *weak,
    uint8_t *linked,
    int *queue
)
{
    memset(linked, 0, DIGIT_PIXELS);

    int head = 0;
    int tail = 0;

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        if (strong[i]) {
            linked[i] = 1;
            queue[tail++] = i;
        }
    }

    if (tail == 0) {
        memcpy(linked, weak, DIGIT_PIXELS);
        return;
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
                if (!linked[nidx] && weak[nidx]) {
                    linked[nidx] = 1;
                    queue[tail++] = nidx;
                }
            }
        }
    }
}

static void binary_dilate3x3(const uint8_t *src, uint8_t *dst)
{
    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            uint8_t value = 0;
            for (int dy = -1; dy <= 1 && !value; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= DIGIT_DIM || ny < 0 || ny >= DIGIT_DIM) {
                        continue;
                    }
                    if (src[idx_2d(nx, ny)]) {
                        value = 1;
                        break;
                    }
                }
            }
            dst[idx_2d(x, y)] = value;
        }
    }
}

static void binary_erode3x3(const uint8_t *src, uint8_t *dst)
{
    for (int y = 0; y < DIGIT_DIM; ++y) {
        for (int x = 0; x < DIGIT_DIM; ++x) {
            uint8_t value = 1;
            for (int dy = -1; dy <= 1 && value; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= DIGIT_DIM || ny < 0 || ny >= DIGIT_DIM) {
                        value = 0;
                        break;
                    }
                    if (!src[idx_2d(nx, ny)]) {
                        value = 0;
                        break;
                    }
                }
            }
            dst[idx_2d(x, y)] = value;
        }
    }
}

static void binary_closing3x3(uint8_t *image, uint8_t *tmp)
{
    binary_dilate3x3(image, tmp);
    binary_erode3x3(tmp, image);
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

    resize_bilinear_to_28(input_gray, width, height, s_ws.resized);

    const float max_radius = ((float)DIGIT_DIM - 1.0f) * 0.5f;
    const float radius = clampf(p->circle_radius, 1.0f, max_radius);
    const float transition = clampf(p->transition_width, 0.5f, radius - 0.5f);

    build_soft_circle_mask(radius, transition, s_ws.mask);

    const float bright_ref = top_bright_mean(
        s_ws.resized,
        p->top_bright_frac,
        s_ws.sort_scratch
    );

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        s_ws.stage1[i] = clampf(
            s_ws.mask[i] * s_ws.resized[i] + (1.0f - s_ws.mask[i]) * bright_ref,
            0.0f,
            1.0f
        );
    }

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        s_ws.darkness[i] = clampf(bright_ref - s_ws.stage1[i], 0.0f, 1.0f);
    }

    gaussian_blur_28(
        s_ws.darkness,
        s_ws.cloud,
        p->cloud_sigma,
        s_ws.blur_tmp
    );

    const float cloud_floor = percentile_masked(
        s_ws.cloud,
        s_ws.mask,
        MASK_REGION_MIN,
        35.0f,
        s_ws.sort_scratch
    );

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        s_ws.cloud_component[i] = clampf(s_ws.cloud[i] - cloud_floor, 0.0f, 1.0f);
    }

    float dark_p90 = percentile_masked(
        s_ws.darkness,
        s_ws.mask,
        MASK_REGION_MIN,
        90.0f,
        s_ws.sort_scratch
    );
    if (dark_p90 < 1e-6f) {
        dark_p90 = 1e-6f;
    }

    const float cloud_strength = p->cloud_strength < 0.0f ? 0.0f : p->cloud_strength;
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        const float protect = 1.0f - clampf(s_ws.darkness[i] / dark_p90, 0.0f, 1.0f);
        const float effective = cloud_strength * protect * s_ws.mask[i];
        const float corrected_darkness = clampf(
            s_ws.darkness[i] - effective * s_ws.cloud_component[i],
            0.0f,
            1.0f
        );
        s_ws.stage2[i] = clampf(bright_ref - corrected_darkness, 0.0f, 1.0f);
    }

    gaussian_blur_28(
        s_ws.stage2,
        s_ws.blur_tmp,
        p->sharpen_radius,
        s_ws.cloud_component
    );

    const float sharpen_amount = p->sharpen_amount < 0.0f ? 0.0f : p->sharpen_amount;
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        const float sharp = clampf(
            s_ws.stage2[i] + sharpen_amount * (s_ws.stage2[i] - s_ws.blur_tmp[i]),
            0.0f,
            1.0f
        );
        s_ws.stage3[i] = clampf(
            s_ws.stage2[i] * (1.0f - s_ws.mask[i]) + sharp * s_ws.mask[i],
            0.0f,
            1.0f
        );
    }

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        s_ws.darkness[i] = clampf(1.0f - s_ws.stage3[i], 0.0f, 1.0f);
    }

    int region_count = collect_masked_values(
        s_ws.darkness,
        s_ws.mask,
        MASK_REGION_MIN,
        s_ws.sort_scratch
    );
    if (region_count <= 0) {
        memcpy(s_ws.sort_scratch, s_ws.darkness, sizeof(float) * DIGIT_PIXELS);
        region_count = DIGIT_PIXELS;
    }

    const float otsu_t = otsu_threshold(s_ws.sort_scratch, region_count);

    float dark_mean = 0.0f;
    float dark_std = 0.0f;
    mean_std_masked(
        s_ws.darkness,
        s_ws.mask,
        MASK_REGION_MIN,
        &dark_mean,
        &dark_std,
        s_ws.sort_scratch
    );

    const float gauss_t = clampf(dark_mean + p->threshold_k * dark_std, 0.0f, 1.0f);
    float threshold = otsu_t < gauss_t ? otsu_t : gauss_t;
    threshold = clampf(threshold, p->threshold_floor, 0.98f);
    const float soft_w = clampf(p->threshold_soft_width, 0.01f, 0.50f);

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        float response = (s_ws.darkness[i] - threshold) / soft_w;
        response = clampf(response, 0.0f, 1.0f);
        response *= s_ws.mask[i];
        s_ws.response[i] = clampf(response, 0.0f, 1.0f);
    }

    const float link_margin = clampf(p->stroke_link_margin, 0.0f, 0.15f);
    const float link_low = clampf(threshold - link_margin, p->threshold_floor, threshold);

    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        const uint8_t in_link_mask = (s_ws.mask[i] >= LINK_MASK_MIN) ? 1 : 0;
        s_ws.strong[i] = (s_ws.darkness[i] >= threshold && in_link_mask) ? 1 : 0;
        s_ws.weak[i] = (s_ws.darkness[i] >= link_low && in_link_mask) ? 1 : 0;
    }

    binary_propagate(s_ws.strong, s_ws.weak, s_ws.linked, s_ws.queue);
    binary_closing3x3(s_ws.linked, s_ws.morph_tmp);

    const float link_den = (threshold - link_low) + 1e-6f;
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        float link_response = (s_ws.darkness[i] - link_low) / link_den;
        link_response = clampf(link_response, 0.0f, 1.0f);
        s_ws.link_response[i] = link_response;

        if (s_ws.linked[i]) {
            const float recovered = 0.70f * link_response;
            if (recovered > s_ws.response[i]) {
                s_ws.response[i] = recovered;
            }
        } else {
            s_ws.response[i] = 0.0f;
        }

        s_ws.response[i] = clampf(s_ws.response[i] * s_ws.mask[i], 0.0f, 1.0f);
    }

    /* Final stage-4 output is inverted polarity: white digit on black background. */
    for (int i = 0; i < DIGIT_PIXELS; ++i) {
        output_28x28[i] = (uint8_t)lroundf(clampf(s_ws.response[i], 0.0f, 1.0f) * 255.0f);
    }

    return ESP_OK;
}
