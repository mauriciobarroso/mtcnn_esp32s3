
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "main.cc"
        "camera.c"
        "models/pnet_model_data.cc"
        "models/rnet_model_data.cc"
        "models/onet_model_data.cc"
        "mtcnn.cc"

    PRIV_REQUIRES console tflite-lib esp32-camera spi_flash
    INCLUDE_DIRS "")
