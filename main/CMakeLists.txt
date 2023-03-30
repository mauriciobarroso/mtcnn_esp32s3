
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "main.cc"
        "models/"
        "utils.cc"

    PRIV_REQUIRES console tflite-lib esp32-camera spi_flash
    INCLUDE_DIRS "")
