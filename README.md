# Face detection with MTCNN, TensorFlow Lite Micro and ESP32-S3
This is an implementation of MTCNN (Multitask Cascading Convolutional Networks) for the ESP32-S3 SoC (System on Chip) using TensorFlow Lite for Microcontrollers. The main goal is to detect faces in low-cost hardware and reduced technical features.

## MTCNN
MTCNN is a framework developed as a solution for both face detection and face alignment. It consist in three stages of convolutional networks that are able to recognize faces and landmark location such as eyes, nose and mouth. 

### P-Net (Proposal Network)
This is a FCN (Fully Convolutional Network) that is used to obtain candidate windows and their bounding box regression vectors. Bounding box regression is a popular technique to predict the localization of boxes when the goal is detecting an object of some pre-defined class. The candidate windows obtained are calibrated with bounding box regression vectors and processed with NMS (Non Max Supression) operator to combine overlapping regions.

<p align="center">
  <img src="images/p-net.webp" alt="P-Net architecture" width="75%"/>
</p>

### R-Net (Refine Network)
The R-Net further reduces the number of candidates, performs calibration with bounding box regression and employs NMS to merge overlapping candidates. This network  is a CNN, not a FCN like P-Net sice there is a dense layer at the last stage of its architecture.

<p align="center">
  <img src="images/r-net.webp" alt="R-Net architecture" width="75%"/>
</p>

### O-Net (Output Network)
This stage is similar to the R-Net, but this Output Network aims to describe the face in more detail and output the five facial landmarks’ positions for eyes, nose and mouth.

<p align="center">
  <img src="images/o-net.webp" alt="O-Net architecture" width="75%"/>
</p>

## TensorFlow implementation
To implement the MTCNN models the tools used were TensorFlow and Google Colab. TensorFlow is an open source library for ML (Machine Learning) developed for Google and it is capable to bulilding and training neural networks to detect patterns and correlations. Google Colab is a product from Google Research and allows write and execute arbitrary python code through the browser, and is specially well suited to ML, data analysis and education.

For a correct implementation of MTCNN, the input and output data of the models must be processed to guarantee the best results. The next diagram shows the diagram block of the pipeline implemented.

<p align="center">
  <img src="images/mtcnn_pipeline.png" alt="MTCNN pipeline" width="50%"/>
</p>

The first step is to perform an image pyramid to create different scales of the input image and detect faces of different sizes. These new scaled images are the inputs to P-Net which generates the offsets and scores for each candidate window. Then these outputs are post-processed to obtain the coordinates where the faces would meet. The R-Net input must be pre-processed with the previous outputs, in this way new candidate windows are obtained. The R-Net outputs are the offsets and scores of the candidate windows that are post-processed to obtain the new coordinates where the faces would meet. Finally, for O-Net, the R-Net process is repeated and the coordinates of the faces in the input image are obtained.

The pre-process consist of two steps, crop the input image according to the bounding boxes coordinates obtained before and resize the cropped images to match the input shape of the model.

<p align="center">
  <img src="images/mtcnn_preprocess.png" alt="MTCNN pre-process" width="25%"/>
</p>

In the other hand, the post-process consist of three steps, apply NMS to combine overlapped regions, calibrate the bounding boxes with the offset obtained before, square and correct the final bounding boxes coordinates.

<p align="center">
  <img src="images/mtcnn_postprocess.png" alt="MTCNN post-process" width="40%"/>
</p>

All the prrocesses detailed before and the models for TensorFlow, TensorFlow Lite and TensorFlow Lite Micro were developed in the next Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mauriciobarroso/mtcnn_esp32s3/blob/main/models_evaluation.ipynb)

## Deploying to ESP32-S3
The last step of model development show the creation of the .c files for all MTCNN models and the .h file for the models settings, which are located in ``main/models/``. The preprocess and postprocess functions that are required for the MTCNN pipeline were implemented in using C/C++ in the files ``utils.cc`` and ``utils.h``, which are located in ``main/``.

The hardware consist in the ESP32-S3-DevKitC-1-N8R8 and a OV2640 camera module. The use of PSRAM is mandatory, instead of ESP32-S3-DevKitC-1-N8R8 another ESP32-S3 with PSRAM can be used.

### 1. Download and install ESP-IDF
This project was developed with ESP-IDF v5.0, so this version or later must be used. The next link contains the necessary instructions to download and install it, do the manual installation.

[Download and install ESP-IDF v5.0](https://docs.espressif.com/projects/esp-idf/en/release-v5.0/esp32s3/get-started/index.html#installation)


### 2. Clone this repository:
```
git clone --recursive https://github.com/mauriciobarroso/mtcnn_esp32s3.git
```

### 3. Configure the project:
Change the values in ``menuconfig->App Configuration->Camera Configuration`` to configure the camera pins and in ``menuconfig->App Configuration->Wi-Fi Configuration`` to set the Wi-Fi credentials.

```
cd mtcnn_esp32s3/
idf.py set-target esp32s3
idf.py menuconfig
```

### 4. Flash and monitor
This project does not need a screen to display the image and the bounding boxes generated by MTCNN. Instead, it uses the console characters to print the output image and other relevant information. To monitor the console output run:

```
idf.py flash monitor
```

The console should print somethig like this:

```
ESP-ROM:esp32s3-20210327
Build:Mar 27 2021
rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)
SPIWP:0xee
mode:DIO, clock div:1
load:0x3fce3810,len:0x17d8
load:0x403c9700,len:0xe88
load:0x403cc700,len:0x3000
entry 0x403c9930
I (25) boot: ESP-IDF v5.0 2nd stage bootloader
I (25) boot: compile time 13:50:38
I (25) boot: chip revision: v0.1
I (26) boot_comm: chip revision: 1, min. bootloader chip revision: 0
I (33) qio_mode: Enabling default flash chip QIO
I (39) boot.esp32s3: Boot SPI Speed : 80MHz
I (44) boot.esp32s3: SPI Mode       : QIO
I (48) boot.esp32s3: SPI Flash Size : 8MB
I (53) boot: Enabling RNG early entropy source...
I (58) boot: Partition Table:
I (62) boot: ## Label            Usage          Type ST Offset   Length
I (69) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (77) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (84) boot:  2 factory          factory app      00 00 00010000 00180000
I (92) boot:  3 storage          Unknown data     01 82 00190000 000f0000
I (99) boot: End of partition table
I (103) boot_comm: chip revision: 1, min. application chip revision: 0
I (111) esp_image: segment 0: paddr=00010020 vaddr=3c0b0020 size=b03d4h (721876) map
I (228) esp_image: segment 1: paddr=000c03fc vaddr=3fc99000 size=05de0h ( 24032) load
I (233) esp_image: segment 2: paddr=000c61e4 vaddr=40374000 size=09e34h ( 40500) load
I (242) esp_image: segment 3: paddr=000d0020 vaddr=42000020 size=a441ch (672796) map
I (345) esp_image: segment 4: paddr=00174444 vaddr=4037de34 size=0b150h ( 45392) load
I (354) esp_image: segment 5: paddr=0017f59c vaddr=50000000 size=00010h (    16) load
I (363) boot: Loaded app from partition at offset 0x10000
I (363) boot: Disabling RNG early entropy source...
I (375) octal_psram: vendor id    : 0x0d (AP)
I (375) octal_psram: dev id       : 0x02 (generation 3)
I (375) octal_psram: density      : 0x03 (64 Mbit)
I (380) octal_psram: good-die     : 0x01 (Pass)
I (385) octal_psram: Latency      : 0x01 (Fixed)
I (390) octal_psram: VCC          : 0x01 (3V)
I (396) octal_psram: SRF          : 0x01 (Fast Refresh)
I (401) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (407) octal_psram: BurstLen     : 0x01 (32 Byte)
I (413) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (419) octal_psram: DriveStrength: 0x00 (1/1)
W (424) PSRAM: DO NOT USE FOR MASS PRODUCTION! Timing parameters will be updated in future IDF version.
I (435) esp_psram: Found 8MB PSRAM device
I (439) esp_psram: Speed: 80MHz
I (443) cpu_start: Pro cpu up.
I (446) cpu_start: Starting app cpu, entry point is 0x40375738
0x40375738: call_start_cpu1 at /home/mauricio/esp/esp-idf-v5.0/components/esp_system/port/cpu_start.c:142

I (0) cpu_start: App cpu up.
I (738) esp_psram: SPI SRAM memory test OK
I (747) cpu_start: Pro cpu start user code
I (747) cpu_start: cpu freq: 240000000 Hz
I (747) cpu_start: Application information:
I (750) cpu_start: Project name:     mtcnn_esp32s3
I (755) cpu_start: App version:      1f8c91f
I (760) cpu_start: Compile time:     Apr 19 2023 14:33:31
I (766) cpu_start: ELF file SHA256:  a41224789833e522...
I (772) cpu_start: ESP-IDF:          v5.0
I (777) heap_init: Initializing. RAM available for dynamic allocation:
I (784) heap_init: At 3FCA5D60 len 000439B0 (270 KiB): D/IRAM
I (791) heap_init: At 3FCE9710 len 00005724 (21 KiB): STACK/DRAM
I (797) heap_init: At 600FE010 len 00001FF0 (7 KiB): RTCRAM
I (804) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (812) spi_flash: detected chip: generic
I (816) spi_flash: flash io: qio
I (820) cpu_start: Starting scheduler on PRO CPU.
I (0) cpu_start: Starting scheduler on APP CPU.
I (841) app: Initializing NVS...
I (861) app: Initializing Wi-Fi...
I (861) pp: pp rom version: e7ae62f
I (861) net80211: net80211 rom version: e7ae62f
I (871) wifi:wifi driver task: 3fcedb6c, prio:23, stack:6656, core=0
I (871) system_api: Base MAC address is not set
I (871) system_api: read default base MAC address from EFUSE
I (891) wifi:wifi firmware version: 0d470ef
I (891) wifi:wifi certification version: v7.0
I (891) wifi:config NVS flash: enabled
I (891) wifi:config nano formating: disabled
I (891) wifi:Init data frame dynamic rx buffer num: 32
I (901) wifi:Init management frame dynamic rx buffer num: 32
I (901) wifi:Init management short buffer num: 32
I (911) wifi:Init dynamic tx buffer num: 32
I (911) wifi:Init tx cache buffer num: 32
I (911) wifi:Init static tx FG buffer num: 2
I (921) wifi:Init static rx buffer size: 1600
I (921) wifi:Init static rx buffer num: 10
I (931) wifi:Init dynamic rx buffer num: 32
I (931) wifi_init: rx ba win: 6
I (931) wifi_init: tcpip mbox: 32
I (941) wifi_init: udp mbox: 6
I (941) wifi_init: tcp mbox: 6
I (951) wifi_init: tcp tx win: 5744
I (951) wifi_init: tcp rx win: 5744
I (951) wifi_init: tcp mss: 1440
I (961) wifi_init: WiFi IRAM OP enabled
I (961) wifi_init: WiFi RX IRAM OP enabled
I (971) phy_init: phy_version 503,13653eb,Jun  1 2022,17:47:08
I (1011) wifi:mode : sta (7c:df:a1:e1:d6:4c)
I (1011) wifi:enable tsf
I (1011) app: Connecting to CASAwifi...
I (1011) app: WIFI_EVENT_STA_START
I (1031) wifi:new:<3,1>, old:<1,0>, ap:<255,255>, sta:<3,1>, prof:1
I (1761) wifi:state: init -> auth (b0)
I (1801) wifi:state: auth -> assoc (0)
I (1851) wifi:state: assoc -> run (10)
I (2011) wifi:connected with CASAwifi, aid = 14, channel 3, 40U, bssid = b0:be:76:04:3e:16
I (2011) wifi:security: WPA2-PSK, phy: bgn, rssi: -44
I (2021) wifi:pm start, type: 1

I (2021) wifi:set rx beacon pti, rx_bcn_pti: 0, bcn_timeout: 0, mt_pti: 25000, mt_time: 10000
I (2021) app: WIFI_EVENT_STA_CONNECTED
I (2141) wifi:BcnInt:102400, DTIM:1
W (3001) wifi:<ba-add>idx:0 (ifx:0, b0:be:76:04:3e:16), tid:0, ssn:6, winSize:64
I (4881) esp_netif_handlers: sta ip: 192.168.1.21, mask: 255.255.255.0, gw: 192.168.1.1
I (4881) app: IP_EVENT_STA_GOT_IP
I (13401) app: Initializing SPIFFS...
I (13451) app: Partition size: total: 896321, used: 0
I (13451) file server: Starting HTTP Server on port: '80'
I (13471) s3 ll_cam: DMA Channel=4
I (13471) cam_hal: cam init ok
I (13471) sccb: pin_sda 4 pin_scl 5
I (13471) sccb: sccb_i2c_port=1
I (13481) camera: Detected camera at address=0x30
I (13481) camera: Detected OV2640 camera
I (13481) camera: Camera PID=0x26 VER=0x42 MIDL=0x7f MIDH=0xa2
I (13561) s3 ll_cam: node_size: 3072, nodes_per_line: 1, lines_per_node: 16
I (13561) s3 ll_cam: dma_half_buffer_min:  3072, dma_half_buffer:  9216, lines_per_half_buffer: 48, dma_buffer_size: 27648
I (13571) cam_hal: buffer_size: 27648, half_buffer_size: 9216, node_buffer_size: 3072, node_cnt: 9, total_cnt: 2
I (13581) cam_hal: Allocating 18432 Byte frame buffer in PSRAM
I (13591) cam_hal: Allocating 18432 Byte frame buffer in PSRAM
I (13601) cam_hal: cam config ok
I (13601) ov2640: Set PLL: clk_2x: 1, clk_div: 3, pclk_auto: 1, pclk_div: 8

```

### 5. Test
Once the device is connected to the previously configured network, point the camera towards any face. The console output should print something like this.
```
P-Net time : 65, bboxes : 3
R-Net time : 232, bboxes : 3
O-Net time : 789, bboxes : 3
MTCNN time : 1088, bboxes : 3
```

To observe the image captured by the camera and processed by MTCNN, you must access the following link http://ip_address/faces.jpg, where "ip_address" must be replaced by the IP address assigned to the device. It should show a result like this.

<p align="center">
  <img src="images/output_faces.jpg" alt="Output faces" width="40%"/>
</p>

## Credits

## License

MIT License

Copyright (c) 2023 Mauricio Barroso Benavides

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
