idf_component_register(
    SRCS
        "main.cc"
        "models/"
        "utils.cc"
        "file_server.c"

    #PRIV_REQUIRES console tflite-lib esp32-camera spi_flash esp_wifi nvs_flash esp_http_server
    INCLUDE_DIRS "")
