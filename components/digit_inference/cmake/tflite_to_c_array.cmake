if(NOT DEFINED INPUT_TFLITE OR INPUT_TFLITE STREQUAL "")
    message(FATAL_ERROR "INPUT_TFLITE is required")
endif()

if(NOT DEFINED OUTPUT_C OR OUTPUT_C STREQUAL "")
    message(FATAL_ERROR "OUTPUT_C is required")
endif()

if(NOT DEFINED OUTPUT_H OR OUTPUT_H STREQUAL "")
    message(FATAL_ERROR "OUTPUT_H is required")
endif()

if(NOT DEFINED SYMBOL_NAME OR SYMBOL_NAME STREQUAL "")
    set(SYMBOL_NAME "g_mnist_tiny_int8_model")
endif()

string(REGEX REPLACE "^\"(.*)\"$" "\\1" INPUT_TFLITE "${INPUT_TFLITE}")
string(REGEX REPLACE "^\"(.*)\"$" "\\1" OUTPUT_C "${OUTPUT_C}")
string(REGEX REPLACE "^\"(.*)\"$" "\\1" OUTPUT_H "${OUTPUT_H}")

if(NOT EXISTS "${INPUT_TFLITE}")
    message(FATAL_ERROR "Input model file does not exist: ${INPUT_TFLITE}")
endif()

get_filename_component(OUTPUT_C_DIR "${OUTPUT_C}" DIRECTORY)
get_filename_component(OUTPUT_H_DIR "${OUTPUT_H}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_C_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_H_DIR}")

file(READ "${INPUT_TFLITE}" MODEL_HEX HEX)
string(LENGTH "${MODEL_HEX}" MODEL_HEX_LEN)

if(MODEL_HEX_LEN EQUAL 0)
    message(FATAL_ERROR "Input model file is empty: ${INPUT_TFLITE}")
endif()

math(EXPR MODEL_BYTE_LEN "${MODEL_HEX_LEN} / 2")

string(TOUPPER "${SYMBOL_NAME}_H_" HEADER_GUARD)
string(REGEX REPLACE "[^A-Z0-9_]" "_" HEADER_GUARD "${HEADER_GUARD}")

get_filename_component(OUTPUT_H_NAME "${OUTPUT_H}" NAME)

set(H_CONTENT
"/* Auto-generated from ${INPUT_TFLITE}. Do not edit manually. */\n"
)
string(APPEND H_CONTENT
"#ifndef ${HEADER_GUARD}\n"
"#define ${HEADER_GUARD}\n\n"
"#include <stddef.h>\n"
"#include <stdint.h>\n\n"
"extern const unsigned char ${SYMBOL_NAME}[];\n"
"extern const size_t ${SYMBOL_NAME}_len;\n\n"
"#endif  /* ${HEADER_GUARD} */\n"
)

set(C_CONTENT
"/* Auto-generated from ${INPUT_TFLITE}. Do not edit manually. */\n"
)
string(APPEND C_CONTENT
"#include \"${OUTPUT_H_NAME}\"\n\n"
"const unsigned char ${SYMBOL_NAME}[] = {\n"
)

set(CURRENT_LINE "    ")
math(EXPR LAST_INDEX "${MODEL_BYTE_LEN} - 1")

if(LAST_INDEX GREATER_EQUAL 0)
    foreach(INDEX RANGE 0 ${LAST_INDEX})
        math(EXPR HEX_POS "${INDEX} * 2")
        string(SUBSTRING "${MODEL_HEX}" ${HEX_POS} 2 BYTE_HEX)
        string(TOLOWER "${BYTE_HEX}" BYTE_HEX)

        string(APPEND CURRENT_LINE "0x${BYTE_HEX}, ")

        math(EXPR COLUMN_MOD "(${INDEX} + 1) % 12")
        if(COLUMN_MOD EQUAL 0)
            string(APPEND C_CONTENT "${CURRENT_LINE}\n")
            set(CURRENT_LINE "    ")
        endif()
    endforeach()
endif()

if(NOT CURRENT_LINE STREQUAL "    ")
    string(APPEND C_CONTENT "${CURRENT_LINE}\n")
endif()

string(APPEND C_CONTENT
"};\n\n"
"const size_t ${SYMBOL_NAME}_len = sizeof(${SYMBOL_NAME});\n"
)

file(WRITE "${OUTPUT_H}" "${H_CONTENT}")
file(WRITE "${OUTPUT_C}" "${C_CONTENT}")

message(STATUS "digit_inference: generated model C array (${MODEL_BYTE_LEN} bytes) from ${INPUT_TFLITE}")