idf_component_register(SRCS "main.c"
                       REQUIRES needle esp_driver_i2s esp_driver_i2c esp_driver_gpio
                                esp_timer fatfs sdmmc esp_driver_sdmmc)

# Per-phase timings (mel, stem, encoder, cross K/V, decoder) are part of what
# this demo prints, so profile the engine in this app.
idf_component_get_property(nd_lib needle COMPONENT_LIB)
target_compile_definitions(${nd_lib} PUBLIC ND_PROFILE)
