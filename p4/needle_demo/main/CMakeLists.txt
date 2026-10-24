# ---------------------------------------------------------------------------
# The tool schema is generated into a header from the canonical JSON file, so
# there is a single source of truth shared with the host tools (the same
# trick as the upstream needle-2 demo; the JSON lives in the untouched
# original/ checkout).
# ---------------------------------------------------------------------------
set(ND_SCHEMA_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../original/needle-2-esp32/tools/led.json")

file(READ "${ND_SCHEMA_PATH}" ND_SCHEMA_RAW)
string(REGEX REPLACE "[\r\n]" "" ND_SCHEMA_RAW "${ND_SCHEMA_RAW}")
string(STRIP "${ND_SCHEMA_RAW}" ND_SCHEMA_RAW)
string(REPLACE "\\" "\\\\" ND_SCHEMA_ESC "${ND_SCHEMA_RAW}")
string(REPLACE "\"" "\\\"" ND_SCHEMA_ESC "${ND_SCHEMA_ESC}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/tools_schema.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/tools_schema.h" @ONLY)

# Re-run CMake when the schema changes so the header never goes stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ND_SCHEMA_PATH}")

idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/generated"
                       REQUIRES needle esp_driver_i2s esp_driver_i2c esp_driver_gpio
                                esp_timer esp_partition fatfs sdmmc esp_driver_sdmmc)

# Per-phase timings (both models) are part of what this demo prints.
idf_component_get_property(nd_lib needle COMPONENT_LIB)
target_compile_definitions(${nd_lib} PUBLIC ND_PROFILE)
