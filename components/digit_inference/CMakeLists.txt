set(DIGIT_INFERENCE_REQUIRES
    esp-tflite-micro
    esp_timer
    esp_jpeg
)

set(STATIC_GEN_DIR "generated_static")
set(PROJECT_MODEL_TFLITE "${CMAKE_SOURCE_DIR}/models/mnist_tiny_int8.tflite")

option(
    DIGIT_INFERENCE_SYNC_MODEL_FROM_TFLITE
    "Generate model C array from models/mnist_tiny_int8.tflite at build time"
    ON
)

if(DIGIT_INFERENCE_SYNC_MODEL_FROM_TFLITE AND EXISTS "${PROJECT_MODEL_TFLITE}")
    set(GENERATED_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_model")
    set(GENERATED_MODEL_C "${GENERATED_MODEL_DIR}/mnist_tiny_model_data.c")
    set(GENERATED_MODEL_H "${GENERATED_MODEL_DIR}/mnist_tiny_model_data.h")
    file(MAKE_DIRECTORY "${GENERATED_MODEL_DIR}")

    add_custom_command(
        OUTPUT "${GENERATED_MODEL_C}" "${GENERATED_MODEL_H}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_MODEL_DIR}"
        COMMAND
            ${CMAKE_COMMAND}
            -DINPUT_TFLITE=${PROJECT_MODEL_TFLITE}
            -DOUTPUT_C=${GENERATED_MODEL_C}
            -DOUTPUT_H=${GENERATED_MODEL_H}
            -DSYMBOL_NAME=g_mnist_tiny_int8_model
            -P "${CMAKE_CURRENT_LIST_DIR}/cmake/tflite_to_c_array.cmake"
        DEPENDS
            "${PROJECT_MODEL_TFLITE}"
            "${CMAKE_CURRENT_LIST_DIR}/cmake/tflite_to_c_array.cmake"
        VERBATIM
        COMMENT "Generating model C array from ${PROJECT_MODEL_TFLITE}"
    )

    set(MODEL_DATA_SOURCE "${GENERATED_MODEL_C}")
    set(MODEL_INCLUDE_DIR "${GENERATED_MODEL_DIR}")
    message(STATUS "digit_inference: model source set to ${PROJECT_MODEL_TFLITE}")
else()
    if(DIGIT_INFERENCE_SYNC_MODEL_FROM_TFLITE)
        message(WARNING
            "digit_inference: model file not found at ${PROJECT_MODEL_TFLITE}; "
            "using checked-in generated_static/mnist_tiny_model_data.c"
        )
    endif()

    set(MODEL_DATA_SOURCE "${STATIC_GEN_DIR}/mnist_tiny_model_data.c")
    set(MODEL_INCLUDE_DIR "${STATIC_GEN_DIR}")
endif()

idf_component_register(
    SRCS
        "digit_inference.cc"
        "digit_preprocess.c"
        "${MODEL_DATA_SOURCE}"
        "${STATIC_GEN_DIR}/digit_test_images_data.c"
    INCLUDE_DIRS
        "include"
        "${MODEL_INCLUDE_DIR}"
        "${STATIC_GEN_DIR}"
    REQUIRES
        ${DIGIT_INFERENCE_REQUIRES}
)

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE -fno-rtti -fno-exceptions)
