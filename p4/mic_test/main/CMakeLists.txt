idf_component_register(SRCS "main.c"
                       REQUIRES esp_driver_i2s esp_driver_i2c esp_driver_gpio
                                esp_timer es8311)
