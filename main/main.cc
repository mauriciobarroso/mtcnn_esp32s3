/**
  ******************************************************************************
  * @file           : main.cc
  * @author         : Mauricio Barroso Benavides
  * @date           : Oct 8, 2022
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * MIT License
  *
  * Copyright (c) 2022 Mauricio Barroso Benavides
  *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
  * of this software and associated documentation files (the "Software"), to
  * deal in the Software without restriction, including without limitation the
  * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  * sell copies of the Software, and to permit persons to whom the Software is
  * furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in
  * all copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  * IN THE SOFTWARE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "camera.h"
#include "mtcnn.h"

#include "models/pnet_model_data.h"
#include "models/rnet_model_data.h"
#include "models/onet_model_data.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Private typedef -----------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
#define SCRATH_BUFFER_SIZE (39 * 1024)
#define TENSOR_ARENA_SIZE (110 * 1024 + SCRATH_BUFFER_SIZE)

/* Private variables ---------------------------------------------------------*/
static const char * TAG = "app"; /* Tag for debugging */

static uint8_t * tensor_arena;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* pnet_scale_1_model = nullptr;
const tflite::Model* pnet_scale_2_model = nullptr;
const tflite::Model * rnet_model = nullptr;
const tflite::Model * onet_model = nullptr;
tflite::MicroInterpreter * interpreter_1 = nullptr;
tflite::MicroInterpreter * interpreter_2 = nullptr;
tflite::MicroInterpreter * rnet_interpreter = nullptr;
tflite::MicroInterpreter * onet_interpreter = nullptr;

/* Private function prototypes -----------------------------------------------*/
static void inference_task(void * arg);

/* Private user code ---------------------------------------------------------*/
extern "C" void app_main() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  /* Map the model into a usable data structure */
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

  /* Reserve memory */
  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", TENSOR_ARENA_SIZE);
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
  		pnet_scale_1_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter_1 = &static_interpreter_1;

  static tflite::MicroInterpreter static_interpreter_2(
  		pnet_scale_2_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter_2 = &static_interpreter_2;

  static tflite::MicroInterpreter static_rnet_interpreter(
  		rnet_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  rnet_interpreter = &static_rnet_interpreter;

  static tflite::MicroInterpreter static_onet_interpreter(
  		onet_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  onet_interpreter = &static_onet_interpreter;

  /* Allocate memory from the tensor_arena for the model's tensors */
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

  /* Initialize Camera */
  int ret = camera_init();
  if (ret != 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "Camera init failed\n");
  }
  TF_LITE_REPORT_ERROR(error_reporter, "Camera Initialized\n");

	/* Create RTOS tasks */
	xTaskCreate(inference_task,
			"Inference Task",
			configMINIMAL_STACK_SIZE * 8,
			NULL,
			tskIDLE_PRIORITY + 8,
			NULL);
}

static void inference_task(void * arg) {
	for(;;) {
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
}

/***************************** END OF FILE ************************************/
