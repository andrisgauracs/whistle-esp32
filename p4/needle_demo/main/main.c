/* Whistle + Needle 2 voice agent on the Waveshare ESP32-P4-ETH: hold the
 * BOOT button, say a request; Whistle transcribes it (TXT), Needle 2 turns
 * the transcript into a grammar-guaranteed tool call (TOK/CONF), and the
 * call is dispatched to a handler that prints an ACT line. Typed requests
 * on the console work exactly the same, minus the microphone.
 *
 * Two models, two memory schemes, one engine:
 *   whistle.cact  16.9 MB  SD card -> PSRAM, runs in place (nd_whistle_*)
 *   needle2.cact  13.7 MB  "needle2" flash partition, memory-mapped in
 *                          place (nd_model_*); KV cache (3.5 MB) in PSRAM
 *
 *   ES8311  I2C num 0: SCL=GPIO8 SDA=GPIO7; PA enable=GPIO53 (kept low)
 *   I2S0    MCK=GPIO13 BCK=GPIO12 WS=GPIO10 DOUT=GPIO9 DIN=GPIO11
 *   button  BOOT = GPIO35 to GND, internal pull-up (0 = pressed)
 *   SD      SDMMC slot 1, 4-bit: CLK=43 CMD=44 D0..D3=39..42, LDO ch.4
 *
 * There is no LED on this board; tool handlers print their ACT line only,
 * so tools/needle_tui.py renders them unchanged.
 *
 * Console, one line each:
 *   in : a request  |  "!think" [0|1]  |  one digit 2..8 = decoder depth
 *   out: EVT ... status/progress, TXT ... transcript, TOK ... streamed call
 *        tokens, CONF x.xxxx, ACT ... tool result, ERR ...
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"
#include "driver/sdmmc_host.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"   /* TF slot VDD = on-chip LDO ch. 4 */
#include "es8311.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "sdmmc_cmd.h"

#include "nd_cact.h"
#include "nd_grammar.h"
#include "nd_ledcmd.h"                     /* nd_led_keywords for STT biasing */
#include "nd_model.h"
#include "nd_quant.h"
#include "nd_sample.h"
#include "nd_tokenizer.h"
#include "nd_whistle.h"

#include "tools_schema.h"

/* Defined in the p4 nd_quant.c overlay: verifies the PIE dot128_s8 kernel
 * against the portable C loop and races the two. */
extern int nd_quant_p4_check(int64_t *pie_us, int64_t *c_us);
extern void nd_quant_p4_init(void);
extern void nd_quant_p4_lut2_enable(void);

static void dot128_selfcheck(const char *app)
{
    int64_t pie_us = 0, c_us = 0;
    int rc = nd_quant_p4_check(&pie_us, &c_us);
    if (rc == 0)
        printf("EVT dot128 pie ok app=%s pie_us=%lld c_us=%lld speedup=%.1fx\n",
               app, (long long)pie_us, (long long)c_us,
               c_us ? (double)c_us / (double)pie_us : 0.0);
    else
        printf("ERR dot128 pie mismatch app=%s vec=%d\n", app, rc);
    fflush(stdout);
}

/* ES8311 codec on I2C num 0 (Waveshare ESP32-P4-ETH). */
#define I2C_NUM        0
#define I2C_SCL_GPIO   8
#define I2C_SDA_GPIO   7
#define PA_GPIO        53            /* speaker amp enable: keep low */

/* I2S num 0. */
#define I2S_MCK_GPIO   13
#define I2S_BCK_GPIO   12
#define I2S_WS_GPIO    10
#define I2S_DO_GPIO    9
#define I2S_DI_GPIO    11

#define BUTTON_GPIO    35            /* BOOT button, active low */

#define MOUNT_POINT    "/sdcard"
#define MODEL_PATH     MOUNT_POINT "/Whistle/whistle.cact"

#define SAMPLE_RATE    16000
#define MAX_SECS       8             /* the cross-attention cache grows with it */
#define CHUNK          160           /* 10 ms per I2S read */
#define DEBOUNCE_MS    20
#define HEAD_DROP_MS   50            /* press transient */
#define TAIL_DROP_MS   30            /* release transient */
#define FLOOR_WIN      320           /* 20 ms noise-floor windows */
#define FLOOR_HIST     250           /* 5 s of idle history */
#define GATE_DB        12.0f         /* speech must clear the floor by this */
#define KW_BONUS       3.0f          /* colour/blink words bias the transcript */

/* Whistle decoder depth at boot; change at runtime with a number key. */
#define WHISTLE_DEPTH  4

/* Needle 2 generation limits (as the S3 demo). */
#define MAX_NEW        96
#define DEFAULT_SECS   2.0f
#define ND_LINE_MAX    256           /* not LINE_MAX: limits.h takes it */

static const char *const LANGS[] = { "en", "de", "fr", "es", "it", "nl", "pl" };

/* The tool schema is generated from tools/led.json at build time; see
 * main/CMakeLists.txt. Edit that JSON, not this file. */
static const char TOOLS_RAW[] = ND_TOOLS_JSON;

/* Compacted at boot: the model was trained on whitespace-free schemas, and
 * an indented one degrades it silently (it starts naming tools that were
 * never declared). Doing this here means the JSON file's formatting is
 * free. */
static char TOOLS_JSON[sizeof(TOOLS_RAW)];

static nd_whistle        s_w;
static nd_model          s_m;
static nd_grammar        s_g;
static sdmmc_card_t     *s_card;
static es8311_handle_t   s_codec;
static i2s_chan_handle_t s_tx, s_rx;
static int               s_mic_slot = -1;   /* 0 = left, 1 = right; -1 = not probed */
static double            s_bench_ms;        /* last bench(), for the status recap */
static int               s_st_ok;      /* Whistle (voice) available */
static int               s_llm_ok;     /* Needle 2 (LLM) available */
static int               s_show_think = 0;   /* "!think" toggles; off = faster */

/* ------------------------------------------------- second-core GEMV worker
 * Shared by both models: every matvec / batched matmul splits its output
 * rows across the two cores (the engine calls nd_parallel_rows). */

static SemaphoreHandle_t s_go, s_done;
static nd_row_fn         s_fn;
static void             *s_ctx;
static uint32_t          s_r0, s_r1;

static void worker_task(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_fn(s_ctx, s_r0, s_r1);
        xSemaphoreGive(s_done);
    }
}

static void rows_dual_core(nd_row_fn fn, void *ctx, uint32_t nrows)
{
    uint32_t half = nrows / 2;
    if (half < 2 || !s_go) {
        fn(ctx, 0, nrows);
        return;
    }
    s_fn  = fn;
    s_ctx = ctx;
    s_r0  = half;
    s_r1  = nrows;
    xSemaphoreGive(s_go);
    fn(ctx, 0, half);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

static void worker_start(void)
{
    s_go   = xSemaphoreCreateBinary();
    s_done = xSemaphoreCreateBinary();
    /* 16 KB: the Hadamard MLP's rows run here too, ~10 KB of scratch deep. */
    xTaskCreatePinnedToCore(worker_task, "nd_worker", 16384, NULL,
                            configMAX_PRIORITIES - 2, NULL, 1);
    nd_parallel_rows = rows_dual_core;
}

/* ------------------------------------------------------------------ codec */

static int codec_init(void)
{
    const i2c_config_t i2c_cfg = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_SDA_GPIO,
        .scl_io_num = I2C_SCL_GPIO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 100000,
    };
    const es8311_clock_config_t clk = {
        .mclk_inverted = false,
        .sclk_inverted = false,
        .mclk_from_mclk_pin = true,    /* MCLK comes from the I2S MCK pin */
        .mclk_frequency = SAMPLE_RATE * 384,
        .sample_frequency = SAMPLE_RATE,
    };

    /* Speaker amp off: no speaker attached on this board. */
    gpio_config_t pa = {
        .pin_bit_mask = 1ULL << PA_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&pa));
    ESP_ERROR_CHECK(gpio_set_level(PA_GPIO, 0));

    ESP_ERROR_CHECK(i2c_param_config(I2C_NUM, &i2c_cfg));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_NUM, I2C_MODE_MASTER, 0, 0, 0));

    s_codec = es8311_create(I2C_NUM, ES8311_ADDRRES_0);
    if (!s_codec) {
        printf("ERR es8311_create (no ACK on I2C? SCL=%d SDA=%d)\n",
               I2C_SCL_GPIO, I2C_SDA_GPIO);
        return -1;
    }
    ESP_ERROR_CHECK(es8311_init(s_codec, &clk, ES8311_RESOLUTION_16,
                                ES8311_RESOLUTION_16));
    ESP_ERROR_CHECK(es8311_sample_frequency_config(s_codec, SAMPLE_RATE * 384,
                                                   SAMPLE_RATE));
    ESP_ERROR_CHECK(es8311_voice_volume_set(s_codec, 70, NULL));
    ESP_ERROR_CHECK(es8311_microphone_config(s_codec, false));
    ESP_ERROR_CHECK(es8311_microphone_gain_set(s_codec, ES8311_MIC_GAIN_24DB));
    return 0;
}

/* ------------------------------------------------------------------- I2S */

static void mic_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0,
                                                            I2S_ROLE_MASTER);
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_MCK_GPIO,
            .bclk = I2S_BCK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_DO_GPIO,
            .din  = I2S_DI_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    chan_cfg.auto_clear = true;           /* tx idles at zeros */
    std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_384;
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;   /* detect the mic slot */

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &s_tx, &s_rx));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

static void button_init(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BUTTON_GPIO,
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
}

/* Which I2S slot carries the mic: count non-zero samples per slot over the
 * first 300 ms (the codec drives one slot). */
static int mic_detect(void)
{
    int16_t raw[2 * CHUNK];
    uint32_t nz[2] = { 0, 0 };
    int chunks;

    for (chunks = 0; chunks < 30; chunks++) {
        size_t got = 0, i, m;
        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got,
                             pdMS_TO_TICKS(200)) != ESP_OK)
            break;
        m = got / (2 * sizeof(int16_t));
        for (i = 0; i < m; i++) {
            nz[0] += (raw[2 * i] != 0);
            nz[1] += (raw[2 * i + 1] != 0);
        }
    }
    s_mic_slot = nz[1] > nz[0] ? 1 : 0;
    printf("EVT mic_channel=%c (nonzero L=%u R=%u)\n",
           s_mic_slot ? 'R' : 'L', (unsigned)nz[0], (unsigned)nz[1]);
    return s_mic_slot;
}

/* --------------------------------------------------------- model sources
 *  - Whistle: SD card -> PSRAM (the whole file read into a heap_caps buffer,
 *    16.9 MB of the 32 MB); flash partition "whistle" as a fallback.
 *  - Needle 2: flash partition "needle2", memory-mapped in place through
 *    the cache - no copy, no PSRAM for weights (the S3 scheme). Trailing
 *    0xFF after the payload is harmless; the engine parses the header. */

static void *model_from_sd(const char *path, size_t *out_size)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 2,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_host_t       host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    FILE              *f;
    long               sz;
    void              *blob = NULL, *chunk = NULL;
    size_t            total = 0;
    int64_t           t0;

    /* The TF slot VDD is wired to the on-chip LDO channel 4 (see the
     * Waveshare 09_sdmmc demo), so it must be enabled first. */
    static sd_pwr_ctrl_handle_t s_pwr;
    sd_pwr_ctrl_ldo_config_t ldo_cfg = { .ldo_chan_id = 4 };
    if (sd_pwr_ctrl_new_on_chip_ldo(&ldo_cfg, &s_pwr) != ESP_OK) {
        printf("ERR sd_ldo\n");
        return NULL;
    }
    host.pwr_ctrl_handle = s_pwr;

    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
    slot.width = 4;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    if (esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot, &mount_cfg,
                                &s_card) != ESP_OK) {
        host.max_freq_khz = SDMMC_FREQ_DEFAULT;   /* retry at 20 MHz */
        if (esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot, &mount_cfg,
                                    &s_card) != ESP_OK) {
            printf("ERR sdcard_mount (card present? seated? FAT formatted?)\n");
            return NULL;
        }
    }
    printf("EVT sdcard mounted\n");
    fflush(stdout);

    f = fopen(path, "rb");
    if (!f) {
        printf("ERR model_open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        printf("ERR model_size %ld\n", sz);
        fclose(f);
        return NULL;
    }
    blob = heap_caps_malloc((size_t)sz, MALLOC_CAP_SPIRAM);
    chunk = heap_caps_malloc(65536, MALLOC_CAP_INTERNAL);
    if (!blob || !chunk) {
        printf("ERR model_alloc bytes=%ld psram_free=%u\n", sz,
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        heap_caps_free(blob);
        heap_caps_free(chunk);
        fclose(f);
        return NULL;
    }
    t0 = esp_timer_get_time();
    while (total < (size_t)sz) {
        size_t want = (size_t)sz - total > 65536 ? 65536 : (size_t)sz - total;
        size_t r = fread(chunk, 1, want, f);
        if (r == 0)
            break;
        memcpy((char *)blob + total, chunk, r);
        total += r;
    }
    fclose(f);
    heap_caps_free(chunk);
    if (total != (size_t)sz) {
        printf("ERR model_read got=%u of %ld\n", (unsigned)total, sz);
        heap_caps_free(blob);
        return NULL;
    }
    printf("EVT model source=sd bytes=%u ms=%.0f MBps=%.1f\n", (unsigned)total,
           (esp_timer_get_time() - t0) / 1000.0,
           total / ((esp_timer_get_time() - t0) / 1000000.0) / (1024.0 * 1024.0));
    fflush(stdout);
    *out_size = (size_t)sz;
    return blob;
}

static const void *model_from_flash(const char *label, size_t *out_size)
{
    const esp_partition_t      *part;
    esp_partition_mmap_handle_t map;
    const void                 *blob;

    part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, label);
    if (!part) {
        printf("ERR model_partition %s (not in the partition table)\n", label);
        return NULL;
    }
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA,
                           &blob, &map) != ESP_OK) {
        printf("ERR model_mmap %s\n", label);
        return NULL;
    }
    printf("EVT model source=flash label=%s bytes=%u\n", label,
           (unsigned)part->size);
    fflush(stdout);
    *out_size = part->size;
    return blob;
}

static const void *whistle_load(size_t *out_size)
{
    const void *blob = model_from_sd(MODEL_PATH, out_size);
    if (blob)
        return blob;
    return model_from_flash("whistle", out_size);   /* unallocated by default */
}

/* Noise floor from the last 5 s of idle audio: the 20th-percentile 20 ms
 * window. The mic runs between presses, so this is known before anyone
 * speaks. */
static float s_floor_hist[FLOOR_HIST];
static int   s_floor_n, s_floor_i;

static float noise_floor(void)
{
    static float tmp[FLOOR_HIST];
    int          i, j, n = s_floor_n;
    if (n < 25)
        return 0.0f;                       /* < 0.5 s seen: let the clip decide */
    memcpy(tmp, s_floor_hist, n * sizeof(float));
    for (i = 1; i < n; i++)
        for (j = i; j > 0 && tmp[j - 1] > tmp[j]; j--) {
            float t = tmp[j]; tmp[j] = tmp[j - 1]; tmp[j - 1] = t;
        }
    return tmp[n / 5];
}

/* ----------------------------------------------------------- Whistle (STT)
 * Transcribes the clip, prints the EVT/TXT lines, and returns the number of
 * tokens (text filled in; may be empty). No LED logic here: the transcript
 * becomes the Needle 2 request. */

static int transcribe(float *x, uint32_t n, char *text, size_t cap)
{
    static const char *PN[ND_WP_COUNT] = { "mel", "stem", "encoder", "cross_kv",
                                           "decoder", "~matmul", "~mlp", "~attn",
                                           "~mhc" };
    uint32_t s0 = 0, s1 = n, T, ids[96], i;
    float    floor_rms = noise_floor(), peak = 0.0f, *enc;
    int      nt;
    int64_t  t0 = esp_timer_get_time(), t_enc, t_dec;

    text[0] = '\0';
    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > peak)
            peak = fabsf(x[i]);
    printf("EVT clip ms=%u peak_dbfs=%.1f floor_dbfs=%.1f\n", (unsigned)(n / 16),
           peak > 0 ? 20.0f * log10f(peak) : -120.0f,
           floor_rms > 0 ? 20.0f * log10f(floor_rms) : -120.0f);

    if (!nd_whistle_find_speech(x, n, floor_rms, GATE_DB, 300, &s0, &s1)) {
        printf("EVT no_speech (nothing %.0f dB above the noise floor)\n", GATE_DB);
        printf("TXT \n");
        fflush(stdout);
        return 0;
    }
    x += s0;
    n  = s1 - s0;
    T  = nd_whistle_frames(&s_w, n);
    printf("EVT speech ms=%u frames=%u\n", (unsigned)(n / 16), (unsigned)T);
    fflush(stdout);

    memset(nd_wprof, 0, sizeof(nd_wprof));
    enc = (float *)heap_caps_malloc((size_t)T * s_w.d * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!enc || nd_whistle_encode(&s_w, x, n, enc) != 0) {
        printf("ERR encode\n");
        heap_caps_free(enc);
        return -1;
    }
    t_enc = esp_timer_get_time();
    nt = nd_whistle_decode(&s_w, enc, T, -1 /* detect */, ids,
                           sizeof(ids) / sizeof(ids[0]));
    t_dec = esp_timer_get_time();
    heap_caps_free(enc);
    if (nt < 0) {
        printf("ERR decode\n");
        return -1;
    }
    nd_whistle_detok(&s_w, ids, (uint32_t)nt, text, cap);

    printf("EVT lang %s\n", LANGS[s_w.lang]);
    printf("TXT %s\n", text);
    printf("EVT time total_ms=%.0f encode_ms=%.0f decode_ms=%.0f tokens=%d audio_ms=%u\n",
           (t_dec - t0) / 1000.0, (t_enc - t0) / 1000.0, (t_dec - t_enc) / 1000.0, nt,
           (unsigned)(n / 16));
    for (i = 0; i < ND_WP_COUNT; i++)
        printf("EVT prof %-9s %8.0f ms\n", PN[i], nd_wprof[i] / 1000.0);
    fflush(stdout);
    return nt;
}

/* ------------------------------------------------------------ argument access
 * The call is grammar-constrained, so it is already well-formed and contains
 * only declared keys - a full JSON parser would be dead weight. */

static int arg_str(const char *json, const char *key, char *out, size_t cap)
{
    char        quoted[ND_GR_STRLEN + 4];
    const char *p;
    size_t      w = 0;

    snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    p = strstr(json, quoted);
    if (!p)
        return 0;
    p = strchr(p + strlen(quoted), ':');
    if (!p)
        return 0;
    p = strchr(p, '"');
    if (!p)
        return 0;
    p++;
    while (*p && *p != '"' && w + 1 < cap)
        out[w++] = *p++;
    out[w] = '\0';
    return w > 0;
}

static float arg_num(const char *json, const char *key, float dflt)
{
    char        quoted[ND_GR_STRLEN + 4];
    const char *p;

    snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    p = strstr(json, quoted);
    if (!p)
        return dflt;
    p = strchr(p + strlen(quoted), ':');
    if (!p)
        return dflt;
    return (float)atof(p + 1);
}

__attribute__((unused))    /* provided for handlers with boolean fields */
static int arg_bool(const char *json, const char *key, int dflt)
{
    char        quoted[ND_GR_STRLEN + 4];
    const char *p;

    snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    p = strstr(json, quoted);
    if (!p)
        return dflt;
    p = strchr(p + strlen(quoted), ':');
    if (!p)
        return dflt;
    return strncmp(p + 1, "true", 4) == 0 || strncmp(p + 2, "true", 4) == 0;
}

/* ===========================================================================
 *                        YOUR TOOLS START HERE
 *
 * To use this project for something else:
 *   1. edit original/needle-2-esp32/tools/led.json  - the constrained schema
 *   2. write a handler below        - receives the call's arguments object
 *   3. add it to the TOOL_TABLE     - matched by the schema's "name"
 *   4. print one ACT line           - the TUI renders it; format is free-form
 *                                     "ACT <what> k=v k=v", or "ACT none"
 * Nothing in engine/ needs to change. (On the S3 the handler also drove a
 * WS2812; this board has no LED, so the ACT line is the whole action.)
 * ======================================================================== */

typedef void (*tool_fn)(const char *args);

static void tool_set_led(const char *args)
{
    char  color[32] = "", mode[16] = "";
    float secs;

    if (!arg_str(args, "color", color, sizeof(color)) ||
        !arg_str(args, "mode", mode, sizeof(mode))) {
        printf("ACT none\n");
        fflush(stdout);
        return;
    }
    secs = arg_num(args, "duration_seconds", DEFAULT_SECS);

    printf("ACT led color=%s mode=%s duration=%.1f\n", color, mode, secs);
    fflush(stdout);
}

static const struct {
    const char *name;
    tool_fn     fn;
} TOOL_TABLE[] = {
    { "set_led", tool_set_led },
};

/* ===========================================================================
 *                         YOUR TOOLS END HERE
 * ======================================================================== */

/* Route a generated tool call to its handler. The grammar guarantees the
 * name is one the schema declared, so an unmatched name means the table and
 * the schema have drifted apart. */
static void dispatch_call(const char *generated)
{
    const char *call = strstr(generated, "<tool_call>");
    const char *args;
    char        name[ND_GR_STRLEN] = "";
    size_t      i;

    if (!call || !arg_str(call, "name", name, sizeof(name))) {
        printf("ACT none\n");          /* the empty call [] - nothing matched */
        fflush(stdout);
        return;
    }

    args = strstr(call, "\"arguments\"");
    for (i = 0; i < sizeof(TOOL_TABLE) / sizeof(TOOL_TABLE[0]); i++) {
        if (!strcmp(TOOL_TABLE[i].name, name)) {
            TOOL_TABLE[i].fn(args ? args : call);
            return;
        }
    }
    printf("ERR no_handler_for %s\n", name);
    fflush(stdout);
}

/* -------------------------------------------------------- Needle 2 (LLM) */

/* Prefill the constant part of the prompt once and snapshot it. Every
 * request then resumes from here, so only the query and the assistant
 * header are prefilled per turn - ~13 tokens instead of ~171. */
static void prime_prefix(void)
{
    static uint32_t ids[512];
    static char     pre[1024];
    int64_t         t0 = esp_timer_get_time();
    int             n;
    int             i;

    snprintf(pre, sizeof(pre), "<|im_start|>user\n<tools>%s</tools>", TOOLS_JSON);

    nd_model_reset(&s_m);
    ids[0] = ND_BOS_ID;
    n = nd_tok_encode(&s_m.tok, pre, strlen(pre), ids + 1,
                      (uint32_t)(sizeof(ids) / sizeof(ids[0]) - 1));
    if (n < 0) {
        printf("ERR prefix_too_long\n");
        return;
    }
    n += 1;
    printf("EVT priming tokens=%d (one-time; every later request reuses this)\n", n);
    fflush(stdout);
    for (i = 0; i < n; i++) {
        nd_model_step_hidden(&s_m, (uint32_t)ids[i]);
        if ((i + 1) % 8 == 0 || i + 1 == n) {
            double el = (esp_timer_get_time() - t0) / 1000000.0;
            double eta = el / (i + 1) * (n - i - 1);
            printf("EVT priming %d/%d  %.0f%%  elapsed=%.0fs  eta=%.0fs\n",
                   i + 1, n, 100.0 * (i + 1) / n, el, eta);
            fflush(stdout);
        }
    }
    nd_model_snapshot(&s_m);
    printf("EVT prefix tokens=%d ms=%.0f sink=%u (cached for all requests)\n",
           n, (esp_timer_get_time() - t0) / 1000.0, (unsigned)s_m.n_sink);
    fflush(stdout);
}

/* Startup benchmark: a fixed number of steps with the KV cache cold, so
 * kernel changes can be measured in seconds instead of running a whole
 * request. */
static void bench(void)
{
    const int N = 6;
    int64_t   t;
    int       i;

    nd_model_rewind(&s_m);
    memset(nd_prof, 0, sizeof(nd_prof));   /* priming already counted into it */
    t = esp_timer_get_time();
    for (i = 0; i < N; i++)
        nd_model_step_hidden(&s_m, (uint32_t)(100 + i));
    {
        double ms = (esp_timer_get_time() - t) / 1000.0;
        s_bench_ms = ms;
        printf("EVT bench tokens=%d ms=%.0f ms_per_tok=%.0f tps=%.3f\n",
               N, ms, ms / N, N / (ms / 1000.0));
#ifdef ND_PROFILE
        {
            static const char *PN[ND_P_COUNT] = {
                "proj2bit", "attention", "hadamard", "mhc_phi4",
                "engram", "logits4", "prep+lut", "confpool" };
            int p;
            for (p = 0; p < ND_P_COUNT; p++)
                printf("EVT prof %-10s %8.1f ms  %5.1f%%\n", PN[p],
                       nd_prof[p] / 1000.0 / N,
                       100.0 * nd_prof[p] / 1000.0 / ms);
        }
#endif
        fflush(stdout);
    }
    nd_model_rewind(&s_m);
}

static void run_inference(const char *query)
{
    static uint32_t ids[512];
    static char     out[1024];
    static char     suf[512];
    nd_sampler      smp;
    const float    *lg = NULL;
    int             n;
    uint32_t        i, produced = 0, w = 0;
    int64_t         t0;
    double          pre_ms;

    snprintf(suf, sizeof(suf),
             "\n%s<|im_end|>\n<|im_start|>assistant\n", query);

    /* Resume from the cached prefix rather than re-running it. */
    nd_model_rewind(&s_m);
    nd_sampler_init(&smp, &s_m.tok, &s_g);

    n = nd_tok_encode_ex(&s_m.tok, suf, strlen(suf), ids,
                         (uint32_t)(sizeof(ids) / sizeof(ids[0])), 0);
    if (n < 0) {
        printf("ERR prompt_too_long\n");
        return;
    }

    t0 = esp_timer_get_time();
    for (i = 0; i < (uint32_t)n; i++) {
        lg = nd_model_step_hidden(&s_m, ids[i]);
        printf("EVT reading %u/%d\n", (unsigned)(i + 1), n);
        fflush(stdout);
    }
    pre_ms = (esp_timer_get_time() - t0) / 1000.0;

    printf("EVT prefill tokens=%d ms=%.0f tps=%.2f sink=%u\n",
           n, pre_ms, n / (pre_ms / 1000.0), (unsigned)s_m.n_sink);
    fflush(stdout);

    /* Optionally skip the reasoning block: force an empty <think></think> and
     * open the call directly. Verified to produce identical calls at roughly
     * half the generated tokens. Default here: off = faster voice replies. */
    if (!s_show_think) {
        uint32_t forced[8], nf = 0, fi, tmp[4];
        int tn;
        forced[nf++] = ND_THINK_START_ID;
        forced[nf++] = ND_THINK_END_ID;
        tn = nd_tok_encode_ex(&s_m.tok, "\n", 1, tmp, 4, 0);
        for (fi = 0; fi < (uint32_t)tn; fi++)
            forced[nf++] = tmp[fi];
        forced[nf++] = ND_TOOL_CALL_START_ID;
        for (fi = 0; fi < nf; fi++) {
            char pc[128];
            nd_sample_accept(&smp, forced[fi]);
            nd_tok_decode_ex(&s_m.tok, &forced[fi], 1, pc, sizeof(pc), 0);
            if (w + strlen(pc) < sizeof(out) - 1) { strcpy(out + w, pc); w += strlen(pc); }
            lg = nd_model_step_hidden(&s_m, forced[fi]);
        }
    }

    t0 = esp_timer_get_time();
    for (i = 0; i < MAX_NEW; i++) {
        uint32_t id = nd_sample_hidden(&s_m, &smp, lg);
        char     piece[256];
        uint32_t k;

        if (id == (uint32_t)-1) {
            printf("ERR no_legal_token\n");
            break;
        }
        if (id == ND_EOS_ID || id == ND_IM_END_ID)
            break;
        nd_sample_accept(&smp, id);

        nd_tok_decode_ex(&s_m.tok, &id, 1, piece, sizeof(piece), 0);
        printf("TOK ");
        for (k = 0; piece[k]; k++) {
            if (piece[k] == '\n')      printf("\\n");
            else if (piece[k] != '\r') putchar(piece[k]);
        }
        printf("\n");
        fflush(stdout);

        if (w + strlen(piece) < sizeof(out) - 1) {
            strcpy(out + w, piece);
            w += strlen(piece);
        }
        produced++;
        lg = nd_model_step_hidden(&s_m, id);
    }
    out[w] = '\0';

    {
        double dec_ms = (esp_timer_get_time() - t0) / 1000.0;
        float  conf   = nd_model_confidence(&s_m);
        if (conf >= 0.0f)
            printf("CONF %.4f\n", conf);
        printf("EVT done tokens=%u ms=%.0f tps=%.2f\n", (unsigned)produced,
               dec_ms, produced ? produced / (dec_ms / 1000.0) : 0.0);
        fflush(stdout);
    }

    dispatch_call(out);
}

/* -------------------------------------------------------------------- main */

/* Re-emit the board summary lines, in the same format as boot.  A TUI (or a
 * human) that attaches after boot has missed them; "s" on the console asks
 * for this recap. */
static void status_reprint(void)
{
    if (s_st_ok)
        printf("EVT ready model=whistle depth=%u enc_layers=%u d_model=%u vocab=%u "
               "psram_free=%u internal_free=%u\n",
               (unsigned)s_w.depth, (unsigned)s_w.enc_layers,
               (unsigned)s_w.d, (unsigned)s_w.vocab,
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    if (s_llm_ok) {
        printf("EVT ready model=needle2 layers=%u d_model=%u vocab=%u "
               "window=%u tools=%u psram_free=%u internal_free=%u\n",
               (unsigned)s_m.n_layers, (unsigned)s_m.d_model,
               (unsigned)s_m.vocab, (unsigned)s_m.window,
               (unsigned)s_g.n_tools,
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        if (s_bench_ms > 0.0)
            printf("EVT bench tokens=%d ms=%.0f ms_per_tok=%.0f tps=%.3f\n", 6,
                   s_bench_ms, s_bench_ms / 6, 6 / (s_bench_ms / 1000.0));
    }
    if (s_mic_slot >= 0)
        printf("EVT mic_channel=%c\n", s_mic_slot ? 'R' : 'L');
    printf("EVT think=%d\n", s_show_think);
    if (s_st_ok)
        printf("EVT depth=%u\n", (unsigned)s_w.depth);
    fflush(stdout);
}

/* One console line arrived (typed on the console or via the TUI). */
static void console_line(const char *line)
{
    size_t len = strlen(line);

    if (len == 0)
        return;

    /* Reasoning control: "!think 0" / "!think 1" set it explicitly, bare
     * "!think" toggles. */
    if (!strncmp(line, "!think", 6)) {
        if (line[6] == ' ')
            s_show_think = (line[7] != '0');
        else
            s_show_think = !s_show_think;
        printf("EVT think=%d\n", s_show_think);
        fflush(stdout);
        return;
    }

    /* Host liveness probe ("p" or "ping"): no side effects. Opening the
     * USB-UART can glitch the board into download mode; host tools send
     * this to check the console is alive before talking. */
    if (!strcmp(line, "p") || !strcmp(line, "ping")) {
        printf("EVT pong\n");
        fflush(stdout);
        return;
    }

    /* Board status recap ("s" for short): re-emit the boot summary lines
     * so a freshly attached TUI can fill its stats panel. */
    if (!strcmp(line, "s") || !strcmp(line, "status")) {
        status_reprint();
        return;
    }

    /* A lone digit 2..9 selects the Whistle decoder depth. */
    if (len == 1 && line[0] >= '2' && line[0] <= '9') {
        uint32_t d = (uint32_t)(line[0] - '0');
        if (s_st_ok && d <= s_w.dec_layers && nd_whistle_set_depth(&s_w, d) == 0)
            printf("EVT depth=%u\n", (unsigned)d);
        else
            printf("ERR depth (2..%u)\n", (unsigned)(s_st_ok ? s_w.dec_layers : 8));
        fflush(stdout);
        return;
    }

    /* Anything else is a request for Needle 2 - the typed twin of a spoken
     * one. */
    if (!s_llm_ok) {
        printf("ERR no_llm (needle2 model not loaded)\n");
        fflush(stdout);
        return;
    }
    run_inference(line);
}

static void agent_task(void *arg)
{
    const uint32_t cap = SAMPLE_RATE * MAX_SECS;
    float         *clip = (float *)heap_caps_malloc(cap * sizeof(float), MALLOC_CAP_SPIRAM);
    int16_t        raw[2 * CHUNK];
    char           line[ND_LINE_MAX];
    int            line_len = 0;
    uint32_t       n = 0;
    int            recording = 0, stable = 1, last = 1;
    int64_t        changed_at = 0, t_start = 0;
    float          win_e = 0.0f;
    uint32_t       win_n = 0;
    (void)arg;

    if (!clip) {
        printf("ERR clip_alloc\n");
        vTaskDelete(NULL);
    }
    if (s_st_ok)
        mic_detect();

    printf("\nEVT ==================================================\n");
    printf("EVT  READY - hold BOOT (GPIO%d) and speak, or type a request\n", BUTTON_GPIO);
    printf("EVT  e.g. \"flash the light blue for two seconds\"\n");
    printf("EVT  console: !think toggles reasoning (now %d)\n", s_show_think);
    if (s_st_ok)
        printf("EVT  console: 2..%u sets the Whistle decoder depth (now %u)\n",
               (unsigned)s_w.dec_layers, (unsigned)s_w.depth);
    printf("EVT ==================================================\n\n");
    fflush(stdout);

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;
        uint8_t c;

        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK)
            continue;
        m = got / (2 * sizeof(int16_t));

        /* Non-blocking console: accumulate a line, dispatch on '\n'. */
        while (read(fileno(stdin), &c, 1) == 1) {
            if (c == '\r')
                continue;
            if (c == '\n') {
                line[line_len] = '\0';
                console_line(line);
                line_len = 0;
            } else if (line_len < ND_LINE_MAX - 1) {
                line[line_len++] = (char)c;
            }
        }

        lvl = gpio_get_level(BUTTON_GPIO);
        if (lvl != last) {
            last = lvl;
            changed_at = esp_timer_get_time();
        } else if (lvl != stable && esp_timer_get_time() - changed_at >= DEBOUNCE_MS * 1000) {
            stable = lvl;
            if (stable == 0) {
                recording = 1;
                n = 0;
                t_start = esp_timer_get_time();
                printf("EVT recording\n");
                fflush(stdout);
            } else if (recording) {
                uint32_t head = HEAD_DROP_MS * 16, tail = TAIL_DROP_MS * 16;
                char     text[512];
                int      nt;
                recording = 0;
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                fflush(stdout);
                if (s_st_ok && n > head + tail + 1600) {
                    nt = transcribe(clip + head, n - head - tail, text, sizeof(text));
                    if (nt > 0 && text[0] && s_llm_ok)
                        run_inference(text);          /* ACT comes from there */
                    else if (nt > 0 && text[0])
                        printf("ERR no_llm (needle2 model not loaded)\nACT none\n");
                    else
                        printf("ACT none\n");
                } else if (s_st_ok) {
                    printf("EVT too_short\nTXT \nACT none\n");
                } else {
                    printf("ERR no_stt (whistle model not loaded)\n");
                }
                fflush(stdout);
                /* Inference stalled the reads; restart the floor estimate on
                 * fresh audio rather than on whatever piled up in the DMA. */
                win_e = 0.0f;
                win_n = 0;
                s_floor_n = 0;
                s_floor_i = 0;
                continue;
            }
        }

        for (i = 0; i < m; i++) {
            float v = (float)raw[2 * i + s_mic_slot] / 32768.0f;
            if (recording) {
                if (n < cap)
                    clip[n++] = v;
            } else {
                win_e += v * v;
                if (++win_n == FLOOR_WIN) {
                    s_floor_hist[s_floor_i] = sqrtf(win_e / FLOOR_WIN);
                    s_floor_i = (s_floor_i + 1) % FLOOR_HIST;
                    if (s_floor_n < FLOOR_HIST)
                        s_floor_n++;
                    win_e = 0.0f;
                    win_n = 0;
                }
            }
        }
        if (recording == 1 && n >= cap) {
            printf("EVT clip_full secs=%d (release to transcribe)\n", MAX_SECS);
            fflush(stdout);
            recording = 2;
        }
    }
}

void app_main(void)
{
    const void *blob;
    size_t      size;

    vTaskDelay(pdMS_TO_TICKS(500));
    /* Non-blocking stdin, for the console lines. */
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
    button_init();
    if (codec_init() != 0)
        return;
    mic_init();
    worker_start();
    nd_quant_p4_init();   /* PSRAM staging mutex for the 2-bit GEMV tensors */
    nd_quant_p4_lut2_enable();  /* PIE lut2 kernel: verify on first gemv */
    dot128_selfcheck("needle_demo");

    if (nd_json_compact(TOOLS_RAW, strlen(TOOLS_RAW),
                        TOOLS_JSON, sizeof(TOOLS_JSON)) < 0) {
        printf("ERR schema_too_large\n");
        return;
    }

    /* ---- Whistle: voice, SD -> PSRAM (flash "whistle" as fallback). */
    blob = whistle_load(&size);
    if (blob && nd_whistle_open(&s_w, blob, size) == 0 &&
        nd_whistle_set_depth(&s_w, WHISTLE_DEPTH) == 0) {
        int k;
        s_st_ok = 1;
        /* Colour and blink words in all seven languages bias the transcript
         * toward the tool vocabulary. No accept() retry: Needle 2 handles
         * free-form requests, so the transcript must not be forced to name
         * a colour in some other language. */
        nd_whistle_set_keywords(&s_w, NULL, 0.0f);
        s_w.kw_bonus = KW_BONUS;
        for (k = 0; k < ND_WH_LANGS; k++)
            nd_whistle_add_keywords(&s_w, k, nd_led_keywords(k));
        printf("EVT ready model=whistle depth=%u enc_layers=%u d_model=%u vocab=%u "
               "psram_free=%u internal_free=%u\n",
               (unsigned)s_w.depth, (unsigned)s_w.enc_layers, (unsigned)s_w.d,
               (unsigned)s_w.vocab,
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    } else {
        printf("ERR whistle_open (voice disabled; typed requests still work)\n");
    }
    fflush(stdout);

    /* ---- Needle 2: flash "needle2" partition, memory-mapped in place. */
    blob = model_from_flash("needle2", &size);
    if (blob) {
        const char *gerr = NULL;
        if (nd_model_open(&s_m, blob, size) != 0) {
            printf("ERR model_open (needle2.cact flashed at 0x410000?)\n");
        } else if (nd_grammar_compile(&s_g, TOOLS_JSON, strlen(TOOLS_JSON),
                                      &gerr) != 0) {
            printf("ERR grammar %s\n", gerr ? gerr : "?");
        } else {
            s_llm_ok = 1;
            printf("EVT ready model=needle2 layers=%u d_model=%u vocab=%u "
                   "window=%u tools=%u psram_free=%u internal_free=%u\n",
                   (unsigned)s_m.n_layers, (unsigned)s_m.d_model,
                   (unsigned)s_m.vocab, (unsigned)s_m.window,
                   (unsigned)s_g.n_tools,
                   (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                   (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
            fflush(stdout);
            prime_prefix();
            bench();
        }
    } else {
        printf("ERR needle2 unavailable (typed/spoken requests disabled)\n");
    }
    fflush(stdout);

    /* The Whistle decoder and the Needle 2 sampler both keep deep scratch;
     * give the task room for either (S3 ran Needle 2 on an 8 KB stack). */
    xTaskCreatePinnedToCore(agent_task, "agent", 48 * 1024, NULL, 5, NULL, 0);
}
