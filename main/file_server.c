/**
  ******************************************************************************
  * @file           : file_server.c
  * @author         : Mauricio Barroso Benavides
  * @date           : Apr 22, 2022
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

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include "esp_http_server.h"
#include "esp_vfs.h"
#include "esp_err.h"
#include "esp_log.h"

/* Private define ------------------------------------------------------------*/
#define SCRATCH_BUFSIZE					(8192)

/* Private macro -------------------------------------------------------------*/
#define FILE_PATH_MAX 					(ESP_VFS_PATH_MAX + CONFIG_SPIFFS_OBJ_NAME_LEN)
#define IS_FILE_EXT(filename, ext)	(strcasecmp(&filename[strlen(filename) - sizeof(ext) + 1], ext) == 0)
#define MIN(X, Y)						(((X) < (Y)) ? (X) : (Y))

/* External variables --------------------------------------------------------*/

/* Private typedef -----------------------------------------------------------*/
struct file_server_data {
    /* Base path of file storage */
    char base_path[ESP_VFS_PATH_MAX + 1];

    /* Scratch buffer for temporary storage during file transfer */
    char scratch[SCRATCH_BUFSIZE];
};

/* Private variables ---------------------------------------------------------*/
/* Tag for debug */
static const char * TAG1 = "file server";

/* Private function prototypes -----------------------------------------------*/
static esp_err_t download_get_handler( httpd_req_t * req );
static const char * get_path_from_uri( char* dest, const char* base_path, const char* uri, size_t dest_size );
static esp_err_t set_content_type_from_file( httpd_req_t* req, const char* filename );

/* Exported functions --------------------------------------------------------*/
esp_err_t file_server_init(void) {
    static struct file_server_data *server_data = NULL;

    if (server_data) {
        ESP_LOGE(TAG1, "File server already started");
        return ESP_ERR_INVALID_STATE;
    }

    /* Allocate memory for server data */
    server_data = (struct file_server_data *)calloc(1, sizeof(struct file_server_data));
    if (!server_data) {
        ESP_LOGE(TAG1, "Failed to allocate memory for server data");
        return ESP_ERR_NO_MEM;
    }
    strlcpy(server_data->base_path, "/spiffs",
            sizeof(server_data->base_path));

    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    /* Use the URI wildcard matching function in order to
     * allow the same handler to respond to multiple different
     * target URIs which match the wildcard scheme */
    config.uri_match_fn = httpd_uri_match_wildcard;

    ESP_LOGI(TAG1, "Starting HTTP Server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG1, "Failed to start file server!");
        return ESP_FAIL;
    }

    /* URI handler for getting uploaded files */
    httpd_uri_t file_download = {
        .uri       = "/*",  // Match all URIs of type /path/to/file
        .method    = HTTP_GET,
        .handler   = download_get_handler,
        .user_ctx  = server_data    // Pass server data as context
    };
    httpd_register_uri_handler(server, &file_download);

    return ESP_OK;
}

/* Private functions ---------------------------------------------------------*/
/* HTTP server handlers and utils */
static esp_err_t download_get_handler( httpd_req_t * req )
{
    char filepath[FILE_PATH_MAX];
    FILE *fd = NULL;
    struct stat file_stat;

    const char *filename = get_path_from_uri(filepath, ((struct file_server_data *)req->user_ctx)->base_path,
                                             req->uri, sizeof(filepath));
    if (!filename) {
        ESP_LOGE(TAG1, "Filename is too long");
        /* Respond with 500 Internal Server Error */
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Filename too long");
        return ESP_FAIL;
    }

    if (stat(filepath, &file_stat) == -1) {
        ESP_LOGE(TAG1, "Failed to stat file : %s", filepath);
        /* Respond with 404 Not Found */
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "File does not exist");
        return ESP_FAIL;
    }

    fd = fopen(filepath, "r");
    if (!fd) {
        ESP_LOGE(TAG1, "Failed to read existing file : %s", filepath);
        /* Respond with 500 Internal Server Error */
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to read existing file");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG1, "Sending file : %s (%ld bytes)...", filename, file_stat.st_size);
    set_content_type_from_file(req, filename);

    /* Retrieve the pointer to scratch buffer for temporary storage */
    char *chunk = ((struct file_server_data *)req->user_ctx)->scratch;
    size_t chunksize;
    do {
        /* Read file in chunks into the scratch buffer */
        chunksize = fread(chunk, 1, SCRATCH_BUFSIZE, fd);

        if (chunksize > 0) {
            /* Send the buffer contents as HTTP response chunk */
            if (httpd_resp_send_chunk(req, chunk, chunksize) != ESP_OK) {
                fclose(fd);
                ESP_LOGE(TAG1, "File sending failed!");
                /* Abort sending file */
                httpd_resp_sendstr_chunk(req, NULL);
                /* Respond with 500 Internal Server Error */
                httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to send file");
               return ESP_FAIL;
           }
        }

        /* Keep looping till the whole file is sent */
    } while (chunksize != 0);

    /* Close file after sending complete */
    fclose(fd);
    ESP_LOGI(TAG1, "File sending complete");

    /* Respond with an empty chunk to signal HTTP response completion */
#ifdef CONFIG_EXAMPLE_HTTPD_CONN_CLOSE_HEADER
    httpd_resp_set_hdr(req, "Connection", "close");
#endif
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

static const char* get_path_from_uri(char *dest, const char *base_path, const char *uri, size_t destsize)
{
    const size_t base_pathlen = strlen(base_path);
    size_t pathlen = strlen(uri);

    const char *quest = strchr(uri, '?');
    if (quest) {
        pathlen = MIN(pathlen, quest - uri);
    }
    const char *hash = strchr(uri, '#');
    if (hash) {
        pathlen = MIN(pathlen, hash - uri);
    }

    if (base_pathlen + pathlen + 1 > destsize) {
        /* Full path string won't fit into destination buffer */
        return NULL;
    }

    /* Construct full path (base + path) */
    strcpy(dest, base_path);
    strlcpy(dest + base_pathlen, uri, pathlen + 1);

    /* Return pointer to path, skipping the base */
    return dest + base_pathlen;
}

static esp_err_t set_content_type_from_file( httpd_req_t * req, const char* filename ) {
	/* content type header */
    if(IS_FILE_EXT(filename, ".pdf"))
    	return httpd_resp_set_type(req, "application/pdf");
    else if(IS_FILE_EXT( filename, ".html"))
       	return httpd_resp_set_type(req, "text/html");
    else if(IS_FILE_EXT(filename, ".jpeg"))
    	return httpd_resp_set_type(req, "image/jpeg");
    else if(IS_FILE_EXT(filename, ".ico"))
    	return httpd_resp_set_type(req, "image/x-icon");
    else if(IS_FILE_EXT(filename, ".css")) {
    	httpd_resp_set_hdr( req, "Content-Encoding", "gzip" );
		return httpd_resp_set_type(req, "text/css");
    }
    else if(IS_FILE_EXT( filename, ".js")) {
    	httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    	return httpd_resp_set_type(req, "application/javascript");
    }
    else if(IS_FILE_EXT( filename, ".map")) {
    	httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    	return httpd_resp_set_type( req, "application/octet-stream");
    }
    else if(IS_FILE_EXT(filename, ".gif"))
    	return httpd_resp_set_type(req, "application/octet-stream");
    else if(IS_FILE_EXT(filename, ".csv"))
    	return httpd_resp_set_type(req, "text/csv");

    /* for any other type always set as plain text */
    return httpd_resp_set_type(req, "text/plain");
}

/***************************** END OF FILE ************************************/
