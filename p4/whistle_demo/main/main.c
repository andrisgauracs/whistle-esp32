/* Whistle on the Waveshare ESP32-P4-ETH: hold the BOOT button, say a
 * command, the transcript comes out of the console.
 *
 * Speech recognition runs entirely on the chip. The full Whistle model
 * (whistle.cact, 16.9 MB) is read from the SD card into the 32 MB hex
 * PSRAM at boot; decoder depth is runtime-selectable (2..dec_layers,
 * default 4 as on the ESP32-S3) with the number keys on the console.
 * If the card is missing, a flash partition "whistle" (data, subtype
 * 0x40) is memory-mapped in place instead - the same flash-resident
 * scheme the needle2 model will use (partition "needle2").
 *
 *   ES8311  I2C num 0: SCL=GPIO8 SDA=GPIO7; PA enable=GPIO53 (kept low)
 *   I2S0    MCK=GPIO13 BCK=GPIO12 WS=GPIO10 DOUT=GPIO9 DIN=GPIO11
 *   button  BOOT = GPIO35 to GND, internal pull-up (0 = pressed)
 *   SD      SDMMC slot 0, 4-bit: CLK=43 CMD=44 D0..D3=39..42
 *
 * There is no LED on this board; the S3 demo's ACT led ... lines are still
 * printed, so tools/needle_tui.py --whistle works unchanged.
 *
 * Out, one line each (the same EVT/ACT vocabulary as the S3 demo):
 *   EVT ...     status, levels and per-phase timings
 *   TXT ...     the transcript
 *   ACT led ... or ACT none
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

#include "nd_ledcmd.h"
#include "nd_quant.h"
#include "nd_whistle.h"

/* Defined in the p4 nd_quant.c overlay: verifies the PIE dot128_s8 kernel
 * against the portable C loop and races the two. */
extern int nd_quant_p4_check(int64_t *pie_us, int64_t *c_us);

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
#define HEAD_DROP_MS   50            /* press transient: the engine normalises to
                                      * the clip peak, so one click would turn
                                      * every word down */
#define TAIL_DROP_MS   30            /* release transient */
#define FLOOR_WIN      320           /* 20 ms noise-floor windows */
#define FLOOR_HIST     250           /* 5 s of idle history */
#define GATE_DB        12.0f         /* speech must clear the floor by this */
#define KW_BONUS       3.0f          /* tuned on 360 clips: tools/eval_whistle.py */

/* Decoder depth at boot; change at runtime with the number keys. The S3
 * demo ran a stripped depth-4 model; here the full model is loaded and the
 * depth just selects the ladder. */
#define WHISTLE_DEPTH  4

static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };
static const char *const LANGS[] = { "en", "de", "fr", "es", "it", "nl", "pl" };
#define N_COLORS (sizeof(COLORS) / sizeof(COLORS[0]))

static nd_whistle        s_w;
static sdmmc_card_t     *s_card;
static es8311_handle_t   s_codec;
static i2s_chan_handle_t s_tx, s_rx;
static int               s_mic_slot;   /* 0 = left, 1 = right */

/* A transcript is a command if it names a colour, in any of the seven
 * languages. When the first language Whistle detects gives none, it tries
 * the next ones (nd_whistle.h, accept). */
static int has_command(const char *text, void *ctx)
{
    nd_led_cmd c;
    (void)ctx;
    return nd_led_parse(text, COLORS, N_COLORS, 3.0f, &c);
}

/* ------------------------------------------------- second-core GEMV worker
 * As needle_demo: every matvec / batched matmul splits its output rows. */

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
 * first 300 ms. The codec drives one slot; the other reads zeros (or
 * crosstalk). mic_test's CLIP ch=L|R lines show the same thing. */
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
 * Two ways to hold a model, one per board resource:
 *  - SD card -> PSRAM: the whole file is read into a heap_caps_malloc'd
 *    buffer. Whistle (16.9 MB) runs here; the 32 MB PSRAM holds it with
 *    room for the runtime state (and later needle2's KV cache).
 *  - flash, memory-mapped in place: the S3 scheme. The blob is flashed
 *    with esptool into a data partition of subtype 0x40 and mapped
 *    read-only through the cache - no copy, no PSRAM. Reserved for
 *    needle2.cact (13.7 MB, partition "needle2"); whistle_demo only
 *    uses it as a fallback when the card is missing. */

/* Mount the card and read the whole model into a PSRAM buffer. */
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

    /* SDMMC_HOST_DEFAULT() selects slot 1 - what the Waveshare demo uses
     * (P4 routes slot pins via the GPIO matrix; TF = clk 43 cmd 44 d0-3 39-42).
     * The TF slot VDD is wired to the on-chip LDO channel 4 (see the Waveshare
     * 09_sdmmc demo), so it must be enabled before the card will answer. */
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
        /* Retry at the conservative 20 MHz before giving up. */
        host.max_freq_khz = SDMMC_FREQ_DEFAULT;
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

/* Map a model straight out of flash (no copy, no PSRAM). Trailing flash
 * after the .cact payload reads 0xFF; the engine parses the header, so
 * padding beyond the file is harmless (the S3 ran the same way). */
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

/* Primary: the SD card copy. Fallback: a "whistle" flash partition
 * (see partitions.csv - not allocated by default; the space is reserved
 * for needle2). */
static const void *model_load(size_t *out_size)
{
    const void *blob = model_from_sd(MODEL_PATH, out_size);
    if (blob)
        return blob;
    return model_from_flash("whistle", out_size);
}

/* Noise floor from the last 5 s of idle audio: the 20th-percentile 20 ms
 * window. The mic runs between presses, so this is known before anyone
 * speaks - better than guessing it from a clip that may be all speech. */
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

/* --------------------------------------------------------------- inference */

static void transcribe(float *x, uint32_t n)
{
    static const char *PN[ND_WP_COUNT] = { "mel", "stem", "encoder", "cross_kv", "decoder",
                                           "~matmul", "~mlp", "~attn", "~mhc" };
    uint32_t s0 = 0, s1 = n, T, ids[96], i;
    float    floor_rms = noise_floor(), peak = 0.0f, *enc;
    int      nt;
    char     text[512];
    int64_t  t0 = esp_timer_get_time(), t_enc, t_dec;
    nd_led_cmd cmd;

    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > peak)
            peak = fabsf(x[i]);
    printf("EVT clip ms=%u peak_dbfs=%.1f floor_dbfs=%.1f\n", (unsigned)(n / 16),
           peak > 0 ? 20.0f * log10f(peak) : -120.0f,
           floor_rms > 0 ? 20.0f * log10f(floor_rms) : -120.0f);

    if (!nd_whistle_find_speech(x, n, floor_rms, GATE_DB, 300, &s0, &s1)) {
        printf("EVT no_speech (nothing %.0f dB above the noise floor)\n", GATE_DB);
        printf("TXT \nACT none\n");
        fflush(stdout);
        return;
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
        return;
    }
    t_enc = esp_timer_get_time();
    nt = nd_whistle_decode(&s_w, enc, T, -1 /* detect */, ids, sizeof(ids) / sizeof(ids[0]));
    t_dec = esp_timer_get_time();
    heap_caps_free(enc);
    if (nt < 0) {
        printf("ERR decode\n");
        return;
    }
    nd_whistle_detok(&s_w, ids, (uint32_t)nt, text, sizeof(text));

    printf("EVT lang %s\n", LANGS[s_w.lang]);
    printf("TXT %s\n", text);
    printf("EVT time total_ms=%.0f encode_ms=%.0f decode_ms=%.0f tokens=%d audio_ms=%u\n",
           (t_dec - t0) / 1000.0, (t_enc - t0) / 1000.0, (t_dec - t_enc) / 1000.0, nt,
           (unsigned)(n / 16));
    for (i = 0; i < ND_WP_COUNT; i++)
        printf("EVT prof %-9s %8.0f ms\n", PN[i], nd_wprof[i] / 1000.0);

    /* No LED on this board: print the command, done. */
    if (nt == 0 || !text[0] || !nd_led_parse(text, COLORS, N_COLORS, 3.0f, &cmd)) {
        printf("ACT none\n");               /* nothing heard, or no colour */
        fflush(stdout);
    } else if (cmd.period > 0.0f) {
        printf("ACT led color=%s mode=flash duration=%.1f period=%.1f\n",
               COLORS[cmd.color], cmd.seconds, cmd.period);
        fflush(stdout);
    } else {
        printf("ACT led color=%s mode=%s duration=%.1f\n", COLORS[cmd.color],
               cmd.flash ? "flash" : "solid", cmd.seconds);
        fflush(stdout);
    }
}

/* -------------------------------------------------------------------- main */

/* Re-emit the ready line (a TUI attaching after boot sends 's' to ask). */
static void status_reprint(void)
{
    printf("EVT ready model=whistle depth=%u enc_layers=%u d_model=%u vocab=%u "
           "psram_free=%u internal_free=%u\n",
           (unsigned)s_w.depth, (unsigned)s_w.enc_layers, (unsigned)s_w.d, (unsigned)s_w.vocab,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    fflush(stdout);
}

static void whistle_task(void *arg)
{
    const uint32_t cap = SAMPLE_RATE * MAX_SECS;
    float         *clip = (float *)heap_caps_malloc(cap * sizeof(float), MALLOC_CAP_SPIRAM);
    int16_t        raw[2 * CHUNK];
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
    mic_detect();

    printf("\nEVT ==================================================\n");
    printf("EVT  READY - hold the BOOT button (GPIO%d) and say a command\n", BUTTON_GPIO);
    printf("EVT  e.g. \"flash the light blue\", \"allume la lumière verte\"\n");
    printf("EVT  languages: en de fr es it nl pl (detected)\n");
    printf("EVT  console: 2..%u set decoder depth (now %u)\n",
           (unsigned)s_w.dec_layers, (unsigned)s_w.depth);
    printf("EVT ==================================================\n\n");
    fflush(stdout);

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;
        uint8_t cmd;

        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK)
            continue;
        m = got / (2 * sizeof(int16_t));

        while (read(fileno(stdin), &cmd, 1) == 1) {
            if (cmd == 'p') {          /* host liveness probe, no side effect */
                printf("EVT pong\n");
                fflush(stdout);
            } else if (cmd == 's') {   /* status recap for a late-attaching TUI */
                status_reprint();
            } else if (cmd >= '2' && cmd <= '9') {
                uint32_t d = cmd - '0';
                if (d <= s_w.dec_layers && nd_whistle_set_depth(&s_w, d) == 0)
                    printf("EVT depth=%u\n", (unsigned)d);
                else
                    printf("ERR depth (2..%u)\n", (unsigned)s_w.dec_layers);
                fflush(stdout);
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
                recording = 0;
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                if (n > head + tail + 1600)
                    transcribe(clip + head, n - head - tail);
                else {
                    printf("EVT too_short\nTXT \nACT none\n");
                }
                fflush(stdout);
                /* Inference stalled the reads; restart the floor estimate on
                 * fresh audio rather than on whatever piled up in the DMA. */
                win_e = 0.0f;
                win_n = 0;
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
    int     rc;

    vTaskDelay(pdMS_TO_TICKS(500));
    /* Non-blocking stdin, for the depth keys. */
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
    button_init();
    if (codec_init() != 0)
        return;
    mic_init();
    worker_start();
    dot128_selfcheck("whistle_demo");

    blob = model_load(&size);
    if (!blob)
        return;
    rc = nd_whistle_open(&s_w, blob, size);
    if (rc != 0) {
        /* -1..-5: no valid .cact; -6: a .cact that is not v4 (e.g.
         * needle2.cact). */
        printf("ERR whistle_open rc=%d (%s on the SD card?)\n", rc,
               MODEL_PATH);
        return;
    }
    if (nd_whistle_set_depth(&s_w, WHISTLE_DEPTH) != 0)
        printf("ERR depth %u (2..%u)\n", WHISTLE_DEPTH, (unsigned)s_w.dec_layers);

    {   /* Colour and blink words for all seven languages; each list only
          * biases a transcript in its own language. */
        int k;
        nd_whistle_set_keywords(&s_w, NULL, 0.0f);
        s_w.kw_bonus = KW_BONUS;
        for (k = 0; k < ND_WH_LANGS; k++)
            nd_whistle_add_keywords(&s_w, k, nd_led_keywords(k));
        s_w.accept = has_command;
    }
    status_reprint();

    /* The decoder keeps a few KB of scratch per call level; give it room. */
    xTaskCreatePinnedToCore(whistle_task, "whistle", 48 * 1024, NULL, 5, NULL, 0);
}
