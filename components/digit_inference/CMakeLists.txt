set(DIGIT_INFERENCE_REQUIRES
    esp-tflite-micro
    esp_timer
    esp_jpeg
)

set(STATIC_GEN_DIR "generated_static")

idf_component_register(
    SRCS
        "digit_inference.cc"
        "digit_preprocess.c"
        "${STATIC_GEN_DIR}/mnist_tiny_model_data.c"
        "${STATIC_GEN_DIR}/digit_test_images_data.c"
    INCLUDE_DIRS
        "include"
        "${STATIC_GEN_DIR}"
    REQUIRES
        ${DIGIT_INFERENCE_REQUIRES}
)

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE -fno-rtti -fno-exceptions)
