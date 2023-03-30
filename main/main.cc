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

#include "sensor.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"
#include "driver/ledc.h"

#include "utils.h"
#include "models/pnet_1.c"
#include "models/pnet_2.c"
#include "models/pnet_3.c"
#include "models/rnet.c"
#include "models/onet.c"

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

tflite::MicroInterpreter * pnet_1_interpreter = nullptr;
tflite::MicroInterpreter * pnet_2_interpreter = nullptr;
tflite::MicroInterpreter * pnet_3_interpreter = nullptr;
tflite::MicroInterpreter * rnet_interpreter = nullptr;
tflite::MicroInterpreter * onet_interpreter = nullptr;

/* Private functions declaration ---------------------------------------------*/
static void tflm_init(void);
static esp_err_t camera_init(void);
static void inference_task(void * arg);

/* Private user code ---------------------------------------------------------*/
extern "C" void app_main() {
	/* Initialize TFLM and allocate tensors for the models */
	tflm_init();

  /* Initialize Camera */
  ESP_ERROR_CHECK(camera_init());

	/* Create RTOS tasks to run inferences */
	xTaskCreate(inference_task,
			"Inference Task",
			configMINIMAL_STACK_SIZE * 8,
			NULL,
			tskIDLE_PRIORITY + 8,
			NULL);
}

/* Private functions definition ----------------------------------------------*/
static void tflm_init(void) {
	tflite::ErrorReporter* error_reporter = nullptr;
  tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  /* Map the model into a usable data structure */
  const tflite::Model * pnet_1_model = tflite::GetModel(pnet_1_model_data);
  if (pnet_1_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 pnet_1_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  const tflite::Model * pnet_2_model = tflite::GetModel(pnet_2_model_data);
  if (pnet_2_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 pnet_2_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  const tflite::Model * pnet_3_model = tflite::GetModel(pnet_3_model_data);
  if (pnet_3_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 pnet_3_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  const tflite::Model * rnet_model = tflite::GetModel(rnet_model_data);
  if (rnet_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 rnet_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  const tflite::Model * onet_model = tflite::GetModel(onet_model_data);
  if (rnet_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
												 onet_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  /* Reserve memory */
  uint8_t * tensor_arena = (uint8_t *) heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

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
  static tflite::MicroInterpreter static_pnet_1_interpreter(
  		pnet_1_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  pnet_1_interpreter = &static_pnet_1_interpreter;

  static tflite::MicroInterpreter static_pnet_2_interpreter(
  		pnet_2_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  pnet_2_interpreter = &static_pnet_2_interpreter;

  static tflite::MicroInterpreter static_pnet_3_interpreter(
    		pnet_3_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    pnet_3_interpreter = &static_pnet_3_interpreter;

  static tflite::MicroInterpreter static_rnet_interpreter(
  		rnet_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  rnet_interpreter = &static_rnet_interpreter;

  static tflite::MicroInterpreter static_onet_interpreter(
  		onet_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  onet_interpreter = &static_onet_interpreter;

  /* Allocate memory from the tensor_arena for the model's tensors */
  TfLiteStatus allocate_status = pnet_1_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  allocate_status = pnet_2_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  allocate_status = pnet_3_interpreter->AllocateTensors();
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
}

static esp_err_t camera_init(void) {
	esp_err_t ret = ESP_OK;

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 11;
  config.pin_d1 = 9;
  config.pin_d2 = 8;
  config.pin_d3 = 10;
  config.pin_d4 = 12;
  config.pin_d5 = 18;
  config.pin_d6 = 17;
  config.pin_d7 = 16;
  config.pin_xclk = 15;
  config.pin_pclk = 13;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_sccb_sda = 4;
  config.pin_sccb_scl = 5;
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.xclk_freq_hz = 15000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = CAM_FRAMESIZE;
  config.jpeg_quality = 10;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  /* Camera initialization */
  ret = esp_camera_init(&config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed");
    return ret;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 1); /* flip it back */

  return ret;
}

static void inference_task(void * arg) {
	for(;;) {
		long long start_time = esp_timer_get_time();

		/* Get image */
	  camera_fb_t* fb = esp_camera_fb_get();
	  if (!fb) {
	    ESP_LOGE(TAG, "Camera capture failed");
	  }

		/* Convert to RGB888 and return frame buffer*/
	  uint8_t * rgb888_image = (uint8_t *)malloc((IMG_W * IMG_H * IMG_CH) * sizeof(uint8_t));
	  fmt2rgb888(fb->buf, fb->len, PIXFORMAT_RGB565, rgb888_image);
	  esp_camera_fb_return(fb);

	  /* Run P-Net for all scales */
	  candidate_windows_t pnet_candidate_windows;
	  pnet_candidate_windows.candidate_window = NULL;
	  pnet_candidate_windows.len = 0;

	  run_pnet(&pnet_candidate_windows, pnet_1_interpreter, rgb888_image, IMG_W, IMG_H, PNET_1_SCALE);
	  run_pnet(&pnet_candidate_windows, pnet_2_interpreter, rgb888_image, IMG_W, IMG_H, PNET_2_SCALE);
	  run_pnet(&pnet_candidate_windows, pnet_3_interpreter, rgb888_image, IMG_W, IMG_H, PNET_3_SCALE);
	  nms(&pnet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

	  bboxes_t * pnet_bboxes;
	  get_calibrated_boxes(&pnet_bboxes, &pnet_candidate_windows);
	  free(pnet_candidate_windows.candidate_window);

	  square_boxes(pnet_bboxes);
	  correct_boxes(pnet_bboxes, IMG_W, IMG_H);

	  long long pnet_time = (esp_timer_get_time() - start_time);
	  printf("Time for P-Net = %lld\n", pnet_time / 1000);
	  printf("Ouput bboxes:%d\n", pnet_bboxes->len);

	  /* Run R-Net */
	  start_time = esp_timer_get_time();

	  candidate_windows_t rnet_candidate_windows;
	  rnet_candidate_windows.candidate_window = NULL;
	  rnet_candidate_windows.len = 0;

	  run_rnet(&rnet_candidate_windows, rnet_interpreter, rgb888_image, IMG_W, IMG_H, pnet_bboxes);
	  free(pnet_bboxes->bbox);
	  free(pnet_bboxes);
	  nms(&rnet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

	  bboxes_t * rnet_bboxes;
	  get_calibrated_boxes(&rnet_bboxes, &rnet_candidate_windows);
	  free(rnet_candidate_windows.candidate_window);

	  square_boxes(rnet_bboxes);
	  correct_boxes(rnet_bboxes, IMG_W, IMG_H);

	  long long rnet_time = (esp_timer_get_time() - start_time);
		printf("Time for R-Net = %lld\n", rnet_time / 1000);
		printf("Ouput bboxes:%d\n", rnet_bboxes->len);

	  /* Run O-Net */
	  start_time = esp_timer_get_time();

	  candidate_windows_t onet_candidate_windows;
	  onet_candidate_windows.candidate_window = NULL;
	  onet_candidate_windows.len = 0;

	  run_onet(&onet_candidate_windows, onet_interpreter, rgb888_image, IMG_W, IMG_H, rnet_bboxes);
	  free(rnet_bboxes->bbox);
	  free(rnet_bboxes);
	  nms(&onet_candidate_windows, NMS_THRESHOLD, IOU_MODE);

	  bboxes_t * onet_bboxes;
	  get_calibrated_boxes(&onet_bboxes, &onet_candidate_windows);
	  free(onet_candidate_windows.candidate_window);

	  square_boxes(onet_bboxes);
	  correct_boxes(onet_bboxes, IMG_W, IMG_H);

	  long long onet_time = (esp_timer_get_time() - start_time);
		printf("Time for O-Net = %lld\n", onet_time / 1000);
		printf("Ouput bboxes:%d\n", onet_bboxes->len);

		/* Print final results */
	  draw_rectangle_rgb888(rgb888_image, onet_bboxes, IMG_W);
	  free(onet_bboxes->bbox);
	  free(onet_bboxes);
	  print_rgb888(rgb888_image, IMG_W, IMG_H);

	  printf("MTCNN time = %lld\n\n", (pnet_time + rnet_time + onet_time) / 1000);

	  free(rgb888_image);
	  vTaskDelay(pdMS_TO_TICKS(20)); /* To avoid watchdog */
	}
}

/***************************** END OF FILE ************************************/
