/* mic_test - hardware check for the Whistle push-to-talk front end on the
 * Waveshare ESP32-P4-ETH.
 *
 * Captures 16 kHz stereo audio from the onboard ES8311 codec over I2S while
 * the BOOT button (GPIO35) is held, and on release prints the clip length
 * plus peak and RMS levels for both slots.
 *
 *   ES8311  I2C num 0: SCL=GPIO8 SDA=GPIO7; PA enable=GPIO53 (kept low:
 *           no speaker attached)
 *   I2S0    MCK=GPIO13 BCK=GPIO12 WS=GPIO10 DOUT=GPIO9 DIN=GPIO11
 *   button  BOOT = GPIO35 to GND, internal pull-up (0 = pressed)
 *
 * The I2S channel runs continuously and samples are only kept while the
 * button is down, so no codec startup transient lands at the head of a clip.
 *
 * Both I2S slots are captured. The codec drives the mic on one slot; the
 * other is a wiring check (see zeros= / jumps= below). The slot with the
 * mic is the one to point whistle_demo at.
 *
 * In (optional): 'p' / 'r' on the console press and release a virtual
 * button, so the capture path can be exercised without touching the board.
 *
 * Out, one line each:
 *   EVT ...                     status
 *   CLIP ch=L|R ms=.. samples=.. peak=.. peak_dbfs=.. rms=.. rms_dbfs=.. dc=..
 *        zeros=..  jumps=..     fraction of exact-zero samples, and of
 *                               sample-to-sample steps over half scale
 *   HB ...                      heartbeat every 5 s: loop is alive
 *   PCM <base64>  ...  PCM end  the clip as interleaved stereo s16le, for
 *                               tools/mic_capture.py
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"
#include "es8311.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

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

#define SAMPLE_RATE    16000
#define MAX_SECS       30
#define CHUNK          160           /* 10 ms frames per read */
#define DEBOUNCE_MS    20
#define HEARTBEAT_MS   5000
#define MCLK_MULTIPLE  I2S_MCLK_MULTIPLE_384
#define MIC_GAIN       ES8311_MIC_GAIN_24DB

static es8311_handle_t s_codec;
static i2s_chan_handle_t s_tx, s_rx;

/* ----------------------------------------------------------------- codec */

static void codec_init(void)
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

    /* Speaker amp off: this check only records. */
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
        return;
    }
    ESP_ERROR_CHECK(es8311_init(s_codec, &clk, ES8311_RESOLUTION_16,
                                ES8311_RESOLUTION_16));
    ESP_ERROR_CHECK(es8311_sample_frequency_config(s_codec, SAMPLE_RATE * 384,
                                                   SAMPLE_RATE));
    ESP_ERROR_CHECK(es8311_voice_volume_set(s_codec, 70, NULL));
    ESP_ERROR_CHECK(es8311_microphone_config(s_codec, false));
    ESP_ERROR_CHECK(es8311_microphone_gain_set(s_codec, MIC_GAIN));
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
    std_cfg.clk_cfg.mclk_multiple = MCLK_MULTIPLE;

    /* Both slots captured: the second one is the wiring check. */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;

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

/* ----------------------------------------------------------------- output */

static double dbfs(double v)
{
    return v > 0.0 ? 20.0 * log10(v / 32768.0) : -120.0;
}

/* Stats for one channel of an interleaved stereo clip of n frames. */
static void report(const int16_t *pcm, size_t n, int ch)
{
    double sum = 0.0, sq = 0.0, mean, rms;
    int    peak = 0;
    size_t i, zeros = 0, jumps = 0;

    if (n == 0) {
        printf("CLIP ch=%c ms=0 samples=0\n", ch ? 'R' : 'L');
        return;
    }
    for (i = 0; i < n; i++) {
        int v = pcm[2 * i + ch], a = abs(v);
        if (a > peak)
            peak = a;
        sum += v;
        zeros += (v == 0);
        /* Real audio never steps more than half scale between samples. */
        if (i && abs(v - pcm[2 * (i - 1) + ch]) > 16384)
            jumps++;
    }
    mean = sum / n;
    /* RMS around the mean, so a DC offset does not floor the "silent"
     * figure. */
    for (i = 0; i < n; i++) {
        double d = pcm[2 * i + ch] - mean;
        sq += d * d;
    }
    rms = sqrt(sq / n);

    printf("CLIP ch=%c ms=%u samples=%u peak=%d peak_dbfs=%.1f rms=%.1f "
           "rms_dbfs=%.1f dc=%.1f zeros=%.3f jumps=%.4f\n",
           ch ? 'R' : 'L', (unsigned)(n * 1000 / SAMPLE_RATE), (unsigned)n,
           peak, dbfs(peak), rms, dbfs(rms), mean,
           (double)zeros / n, (double)jumps / n);
}

static void dump_pcm(const int16_t *pcm, size_t n)
{
    static const char B64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t *p   = (const uint8_t *)pcm;
    size_t         len = 2 * n * sizeof(int16_t), i;
    char           line[4 + 128 + 2];
    int            k = 0;

    /* 96 input bytes -> 128 base64 chars per line. */
    for (i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        int      m = (int)(len - i);

        if (m > 1) v |= (uint32_t)p[i + 1] << 8;
        if (m > 2) v |= p[i + 2];
        line[k++] = B64[(v >> 18) & 63];
        line[k++] = B64[(v >> 12) & 63];
        line[k++] = m > 1 ? B64[(v >> 6) & 63] : '=';
        line[k++] = m > 2 ? B64[v & 63] : '=';
        if (k == 128 || i + 3 >= len) {
            line[k] = '\0';
            printf("PCM %s\n", line);
            k = 0;
            /* 1 ms per 128-char line, ~128 KB/s: slow enough that the host's
             * polling keeps up and the driverless console drops nothing.
             * Also lets the idle task run during a long dump. */
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    printf("PCM end rate=%d ch=2\n", SAMPLE_RATE);
}

/* ---------------------------------------------------------------- console */

/* The console stays on the default driverless USB-Serial/JTAG path, which
 * never blocks: it drops bytes if the host is slow to poll. Paced dumps and
 * a host that tolerates bad lines (tools/mic_capture.py) handle that. */
static void console_init(void)
{
    /* Non-blocking stdin, for the 'p' / 'r' virtual button. */
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
}

/* ------------------------------------------------------------------- main */

void app_main(void)
{
    const size_t cap = (size_t)SAMPLE_RATE * MAX_SECS;
    int16_t     *clip;
    int16_t      raw[2 * CHUNK];
    size_t       n = 0;
    int          recording = 0, stable = 1, last = 1, vpress = 0;
    int64_t      changed_at = 0, t_start = 0, t_hb = 0;
    unsigned     reads = 0, timeouts = 0;

    vTaskDelay(pdMS_TO_TICKS(500));
    console_init();
    button_init();
    codec_init();
    mic_init();

    clip = heap_caps_malloc(2 * cap * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!clip) {
        printf("ERR clip_alloc bytes=%u\n", (unsigned)(2 * cap * sizeof(int16_t)));
        return;
    }
    printf("EVT mic_test ready rate=%d max_secs=%d psram_free=%u\n",
           SAMPLE_RATE, MAX_SECS,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("EVT hold the BOOT button (GPIO%d) and speak\n", BUTTON_GPIO);
    fflush(stdout);

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;
        uint8_t cmd;

        /* Blocks for one 10 ms DMA frame; this also paces the loop. A
         * timeout here means the I2S clock or DMA has stopped. */
        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got,
                             pdMS_TO_TICKS(500)) != ESP_OK) {
            printf("ERR i2s_timeout count=%u\n", ++timeouts);
            fflush(stdout);
            continue;
        }
        reads++;
        m = got / (2 * sizeof(int16_t));         /* stereo frames */

        while (read(fileno(stdin), &cmd, 1) == 1) {
            if (cmd == 'p') vpress = 1;
            if (cmd == 'r') vpress = 0;
        }
        lvl = vpress ? 0 : gpio_get_level(BUTTON_GPIO);

        if (esp_timer_get_time() - t_hb >= HEARTBEAT_MS * 1000) {
            t_hb = esp_timer_get_time();
            printf("HB reads=%u timeouts=%u button=%d stable=%d rec=%d\n",
                   reads, timeouts, gpio_get_level(BUTTON_GPIO), stable,
                   recording);
            fflush(stdout);
        }
        if (lvl != last) {
            last       = lvl;
            changed_at = esp_timer_get_time();
        } else if (lvl != stable &&
                   esp_timer_get_time() - changed_at >= DEBOUNCE_MS * 1000) {
            stable = lvl;
            if (stable == 0) {
                recording = 1;
                n         = 0;
                t_start   = esp_timer_get_time();
                printf("EVT recording\n");
                fflush(stdout);
            } else if (recording) {
                recording = 0;
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                report(clip, n, 0);
                report(clip, n, 1);
                dump_pcm(clip, n);
                fflush(stdout);
            }
        }

        if (recording) {
            for (i = 0; i < m && n < cap; i++, n++) {
                clip[2 * n]     = raw[2 * i];
                clip[2 * n + 1] = raw[2 * i + 1];
            }
            if (n == cap && recording == 1) {
                printf("EVT clip_full secs=%d (release to finish)\n", MAX_SECS);
                fflush(stdout);
                recording = 2;      /* stop appending */
            }
        }
    }
}
