set(DIGIT_INFERENCE_REQUIRES
    esp-tflite-micro
    esp_timer
)

if(CMAKE_SCRIPT_MODE_FILE)
    idf_component_register(
        SRCS
            "digit_inference.cc"
            "digit_preprocess.c"
        INCLUDE_DIRS
            "include"
        REQUIRES
            ${DIGIT_INFERENCE_REQUIRES}
    )
else()
    idf_build_get_property(PROJECT_DIR PROJECT_DIR)

    if(NOT DEFINED PYTHON OR "${PYTHON}" STREQUAL "")
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(PYTHON "${Python3_EXECUTABLE}")
    endif()

    set(MODEL_INPUT "${PROJECT_DIR}/models/mnist_tiny_int8.tflite")
    set(MODEL_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(MODEL_C "${MODEL_GEN_DIR}/mnist_tiny_model_data.c")
    set(MODEL_H "${MODEL_GEN_DIR}/mnist_tiny_model_data.h")

    # ESP-IDF validates include dirs at configure time, so ensure this exists.
    file(MAKE_DIRECTORY "${MODEL_GEN_DIR}")

    add_custom_command(
        OUTPUT "${MODEL_C}" "${MODEL_H}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MODEL_GEN_DIR}"
        COMMAND "${PYTHON}"
                "${PROJECT_DIR}/scripts/tflite_to_c_array.py"
                --input "${MODEL_INPUT}"
                --output-c "${MODEL_C}"
                --output-h "${MODEL_H}"
                --symbol "g_mnist_tiny_int8_model"
        DEPENDS "${MODEL_INPUT}" "${PROJECT_DIR}/scripts/tflite_to_c_array.py"
        COMMENT "Generating mnist_tiny_int8 model source"
        VERBATIM
    )

    add_custom_target(generate_mnist_tiny_model_data DEPENDS "${MODEL_C}" "${MODEL_H}")

    idf_component_register(
        SRCS
            "digit_inference.cc"
            "digit_preprocess.c"
            "${MODEL_C}"
        INCLUDE_DIRS
            "include"
            "${MODEL_GEN_DIR}"
        REQUIRES
            ${DIGIT_INFERENCE_REQUIRES}
    )

    add_dependencies(${COMPONENT_LIB} generate_mnist_tiny_model_data)

    target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
    target_compile_options(${COMPONENT_LIB} PRIVATE -fno-rtti -fno-exceptions)
endif()
