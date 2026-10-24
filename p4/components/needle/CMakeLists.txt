# Shared needle engine component for the p4 apps.
#
# Builds every engine source from original/whistle-esp32/engine (the
# superset engine that runs both whistle.cact and needle2.cact) EXCEPT
# nd_quant.c, which is replaced by the overlay copy in src/ below that
# adds an ESP32-P4 PIE (xesppie/xespv) implementation of dot128_s8 plus
# a boot-time self check. The originals are never modified.
set(ND_ENGINE "${CMAKE_CURRENT_LIST_DIR}/../../../original/whistle-esp32/engine")

file(GLOB ND_SRC "${ND_ENGINE}/src/*.c")
list(FILTER ND_SRC EXCLUDE REGEX "nd_quant\\.c$")

idf_component_register(SRCS ${ND_SRC}
                       src/nd_quant.c
                       INCLUDE_DIRS "${ND_ENGINE}/include"
                       REQUIRES esp_timer heap esp_common)
