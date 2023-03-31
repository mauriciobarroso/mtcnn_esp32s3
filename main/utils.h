/**
  ******************************************************************************
  * @file           : mtcnn.h
  * @author         : Mauricio Barroso Benavides
  * @date           : Oct 2, 2022
  * @brief          : todo: write brief 
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef MTCNN_H_
#define MTCNN_H_

#include "tensorflow/lite/micro/micro_interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "models/models_settings.h"

/* Exported macro ------------------------------------------------------------*/
#define PNET_SIZE				12
#define RNET_SIZE				24
#define ONET_SIZE				48

#define PNET_THRESHOLD  0.5
#define RNET_THRESHOLD  0.7
#define ONET_THRESHOLD  0.8

#define NMS_THRESHOLD   0.35

#define IMG_MIN(A, B) ((A) < (B) ? (A) : (B))
#define IMG_MAX(A, B) ((A) < (B) ? (B) : (A))

/* Exported typedef ----------------------------------------------------------*/
typedef enum {
    MIN_MODE = 0,
    IOU_MODE
} nms_mode_t;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} coordinates_t;

typedef struct {
    float score;
    coordinates_t window;
    coordinates_t offsets;
} candidate_window_t;

typedef struct {
    uint8_t len;
    candidate_window_t * candidate_window;
} candidate_windows_t;

typedef struct {
    uint8_t len;
    coordinates_t * bbox;
} bboxes_t;

/* Exported variables --------------------------------------------------------*/

/* Exported functions prototypes ---------------------------------------------*/
void print_candidate_windows(candidate_windows_t * candidate_windows);
void get_pnet_boxes(bboxes_t ** bboxes, uint16_t width, uint16_t height, float scale);
void add_candidate_windows(candidate_windows_t * candidate_windows, float * probs, float * offsets, bboxes_t * bboxes, float threshold);
void nms(candidate_windows_t * candidate_windows, float threshold, nms_mode_t mode);
void print_bboxes(bboxes_t * bboxes);
void get_calibrated_boxes(bboxes_t ** bboxes, candidate_windows_t * candidate_windows);
void square_boxes(bboxes_t * bboxes);
void correct_boxes(bboxes_t * bboxes, uint16_t w, uint16_t h);
void draw_rectangle_rgb888(uint8_t * buf, bboxes_t * bboxes, int width);
void print_rgb888(uint8_t * img, int width, int height);
void crop_rgb888_img(uint8_t * src, uint8_t * dst, uint16_t width, coordinates_t * coordinates);
void image_zoom_in_twice(uint8_t * dimage, int dw, int dh, int dc, uint8_t * simage, int sw,int sc);
void image_resize_linear(uint8_t * dst_image, uint8_t * src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h);

/* MTCNN */
void run_pnet(candidate_windows_t * candidate_windows, tflite::MicroInterpreter * interpreter, uint8_t * img, uint16_t w, uint16_t h, float scale);
void run_rnet(candidate_windows_t * candidate_windows, tflite::MicroInterpreter * interpreter, uint8_t * img, uint16_t w, uint16_t h, bboxes_t * bboxes);
void run_onet(candidate_windows_t * candidate_windows, tflite::MicroInterpreter * interpreter, uint8_t * img, uint16_t w, uint16_t h, bboxes_t * bboxes);

#ifdef __cplusplus
}
#endif

#endif /* MTCNN_H_ */

/***************************** END OF FILE ************************************/
