/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models/pnet_model_data.h"
#include "models/rnet_model_data.h"
#include "models/onet_model_data.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "app_camera_esp.h"
#include "esp_camera.h"

#include "esp_main.h"
#include "mtcnn.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* pnet_scale_1_model = nullptr;
const tflite::Model* pnet_scale_2_model = nullptr;
const tflite::Model * rnet_model = nullptr;
const tflite::Model * onet_model = nullptr;
tflite::MicroInterpreter * interpreter_1 = nullptr;
tflite::MicroInterpreter * interpreter_2 = nullptr;
tflite::MicroInterpreter * rnet_interpreter = nullptr;
tflite::MicroInterpreter * onet_interpreter = nullptr;
TfLiteTensor * input_1 = nullptr;
TfLiteTensor * input_2 = nullptr;
TfLiteTensor * rnet_input = nullptr;
TfLiteTensor * onet_input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.
constexpr int scratchBufSize = 39 * 1024;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 110 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  pnet_scale_1_model = tflite::GetModel(pnet_1_model_data);
  if (pnet_scale_1_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 pnet_scale_1_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  pnet_scale_2_model = tflite::GetModel(pnet_2_model_data);
  if (pnet_scale_2_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 pnet_scale_2_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  rnet_model = tflite::GetModel(rnet_model_data);
  if (rnet_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 rnet_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  onet_model = tflite::GetModel(onet_model_data);
  if (rnet_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 onet_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  /* Pull in only the operation implementations we need. This relies on a
   * complete list of all the ops needed by this graph. An easier approach is to
   * just use the AllOpsResolver, but this will incur some penalty in code space
   * for op implementations that are not needed by this graph
   *
   */
  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddPrelu();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddTranspose();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  /* Build an interpreter to run the model with */
  static tflite::MicroInterpreter static_interpreter_1(
  		pnet_scale_1_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter_1 = &static_interpreter_1;

  static tflite::MicroInterpreter static_interpreter_2(
  		pnet_scale_2_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter_2 = &static_interpreter_2;

  static tflite::MicroInterpreter static_rnet_interpreter(
  		rnet_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  rnet_interpreter = &static_rnet_interpreter;

  static tflite::MicroInterpreter static_onet_interpreter(
  		onet_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  onet_interpreter = &static_onet_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter_1->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  allocate_status = interpreter_2->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  allocate_status = rnet_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  allocate_status = onet_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input_1 = interpreter_1->input(0);
  input_2 = interpreter_2->input(0);
  rnet_input = rnet_interpreter->input(0);
  onet_input = onet_interpreter->input(0);

  /* Initialize Camera */
  int ret = app_camera_init();
  if (ret != 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "Camera init failed\n");
  }
  TF_LITE_REPORT_ERROR(error_reporter, "Camera Initialized\n");
}

void loop() {
	long long start_time = esp_timer_get_time();

	/* Get image */
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE("camera", "Camera capture failed");
  }

	/* Convert to RGB888 and return frame buffer*/
  uint8_t * rgb888_image = (uint8_t *)malloc((IMG_WIDTH * IMG_HEIGHT * 3) * sizeof(uint8_t));
  fmt2rgb888(fb->buf, fb->len, PIXFORMAT_RGB565, rgb888_image);
  esp_camera_fb_return(fb);

  /* Run P-Net for all scales */
  candidate_windows_t pnet_candidate_windows;
  pnet_candidate_windows.candidate_window = NULL;
  pnet_candidate_windows.len = 0;

  run_pnet(&pnet_candidate_windows, interpreter_1, rgb888_image, IMG_WIDTH, IMG_HEIGHT, SCALE_1);
  run_pnet(&pnet_candidate_windows, interpreter_2, rgb888_image, IMG_WIDTH, IMG_HEIGHT, SCALE_2);
  nms(&pnet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

  bboxes_t * pnet_bboxes;
  get_calibrated_boxes(&pnet_bboxes, &pnet_candidate_windows);
  free(pnet_candidate_windows.candidate_window);

  square_boxes(pnet_bboxes);
  correct_boxes(pnet_bboxes, IMG_WIDTH, IMG_HEIGHT);

  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time for P-Net = %lld\n", total_time / 1000);
  printf("Ouput bboxes:%d\n", pnet_bboxes->len);

  /* Run R-Net */
  start_time = esp_timer_get_time();

  candidate_windows_t rnet_candidate_windows;
  rnet_candidate_windows.candidate_window = NULL;
  rnet_candidate_windows.len = 0;

  run_rnet(&rnet_candidate_windows, rnet_interpreter, rgb888_image, IMG_WIDTH, IMG_HEIGHT, pnet_bboxes);
  free(pnet_bboxes->bbox);
  free(pnet_bboxes);
  nms(&rnet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

  bboxes_t * rnet_bboxes;
  get_calibrated_boxes(&rnet_bboxes, &rnet_candidate_windows);
  free(rnet_candidate_windows.candidate_window);

  square_boxes(rnet_bboxes);
  correct_boxes(rnet_bboxes, IMG_WIDTH, IMG_HEIGHT);

	total_time = (esp_timer_get_time() - start_time);
	printf("Total time for R-Net = %lld\n", total_time / 1000);
	printf("Ouput bboxes:%d\n", rnet_bboxes->len);

  /* Run O-Net */
  start_time = esp_timer_get_time();

  candidate_windows_t onet_candidate_windows;
  onet_candidate_windows.candidate_window = NULL;
  onet_candidate_windows.len = 0;

  run_onet(&onet_candidate_windows, onet_interpreter, rgb888_image, IMG_WIDTH, IMG_HEIGHT, rnet_bboxes);
  free(rnet_bboxes->bbox);
  free(rnet_bboxes);
  nms(&onet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

  bboxes_t * onet_bboxes;
  get_calibrated_boxes(&onet_bboxes, &onet_candidate_windows);
  free(onet_candidate_windows.candidate_window);

  square_boxes(onet_bboxes);
  correct_boxes(onet_bboxes, IMG_WIDTH, IMG_HEIGHT);

	total_time = (esp_timer_get_time() - start_time);
	printf("Total time for O-Net = %lld\n", total_time / 1000);
	printf("Ouput bboxes:%d\n", onet_bboxes->len);

	/* Print final results */
  draw_rectangle_rgb888(rgb888_image, onet_bboxes, IMG_WIDTH);
  free(onet_bboxes->bbox);
  free(onet_bboxes);
  print_rgb888(rgb888_image, IMG_WIDTH, IMG_HEIGHT);


  free(rgb888_image);
  vTaskDelay(pdMS_TO_TICKS(20)); /* To avoid watchdog */
}
