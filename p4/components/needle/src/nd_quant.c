/* ---------------------------------------------------------------------------
 * P4 OVERLAY COPY of original/whistle-esp32/engine/src/nd_quant.c.
 *
 * The p4/components/needle component builds this file in place of the
 * engine's own nd_quant.c. It is a byte-for-byte copy plus:
 *   - a dot128_s8 branch for the ESP32-P4 HP core using the xesppie/xespv
 *     "PIE" DSP instructions (128-bit q registers + xacc accumulator), and
 *   - nd_quant_p4_check(), a boot-time asm-vs-C self check with timings.
 * Keep everything else in sync with the original file (a diff of the two
 * should show only those additions).
 * ------------------------------------------------------------------------ */
#include "nd_quant.h"

#include <math.h>
#include <stdlib.h>
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#include "esp_timer.h"
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#endif
/* Decoded rows and the byte table are read once per multiply-add: keep them
 * in internal SRAM, not PSRAM. */
#define ND_QUANT_ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define ND_QUANT_FREE(p)  heap_caps_free(p)
#else
#define ND_QUANT_ALLOC(n) malloc(n)
#define ND_QUANT_FREE(p)  free(p)
#endif

#include <math.h>
#include <string.h>

float nd_f16_slow(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (man == 0) {
            bits = sign; /* +-0 */
        } else {
            /* Subnormal: renormalise into a float32 exponent. */
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13); /* inf / nan */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }

    {
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
}

ND_HOT void nd_fwht(float *x, uint32_t n)
{
    uint32_t len;

    for (len = 1; len < n; len <<= 1) {
        uint32_t i;
        for (i = 0; i < n; i += len << 1) {
            uint32_t j;
            for (j = i; j < i + len; j++) {
                float a = x[j];
                float b = x[j + len];
                x[j]       = a + b;
                x[j + len] = a - b;
            }
        }
    }
}

uint32_t nd_cq_scratch(const nd_tensor *t)
{
    return nd_cq_in_pad(t);
}

void nd_cq_prepare(const nd_tensor *t, const float *x, float *xh)
{
    uint32_t in_pad = nd_cq_in_pad(t);
    uint32_t g      = t->group;
    uint32_t ngroup = in_pad / g;
    float    scale  = 1.0f / sqrtf((float)g);
    uint32_t gi;

    memcpy(xh, x, t->shape[1] * sizeof(float));
    if (in_pad > t->shape[1])
        memset(xh + t->shape[1], 0, (in_pad - t->shape[1]) * sizeof(float));

    for (gi = 0; gi < ngroup; gi++) {
        float   *blk = xh + (size_t)gi * g;
        uint32_t j;
        nd_fwht(blk, g);
        for (j = 0; j < g; j++)
            blk[j] *= scale;
    }
}

typedef struct {
    const uint8_t  *packed;
    const uint16_t *norms;
    const float    *xh, *cb;
    float          *y;
    uint32_t        ngroup, g, bits, rowbytes, base;
} gemv_ctx;

/* Sum over one group of cb[idx] * xh[j], for the common packings.
 * Indices are an LSB-first bitstream per row: index k occupies bits
 * [k*bits, (k+1)*bits). */
static ND_HOT float dot_group(const uint8_t *p, uint32_t bit_off, uint32_t bits,
                       uint32_t g, const float *cb, const float *xh)
{
    float    s = 0.0f;
    uint32_t j;

    /* Four independent accumulators: a single running sum serialises on the
     * FPU's add latency, which dominates this loop on an LX7. */
    if (bits == 2 && (bit_off & 7u) == 0) {
        const uint8_t *q = p + (bit_off >> 3);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (j = 0; j < g; j += 8) {
            uint8_t b0 = q[0], b1 = q[1];
            q += 2;
            s0 += cb[b0 & 3u]        * xh[j];
            s1 += cb[(b0 >> 2) & 3u] * xh[j + 1];
            s2 += cb[(b0 >> 4) & 3u] * xh[j + 2];
            s3 += cb[(b0 >> 6) & 3u] * xh[j + 3];
            s0 += cb[b1 & 3u]        * xh[j + 4];
            s1 += cb[(b1 >> 2) & 3u] * xh[j + 5];
            s2 += cb[(b1 >> 4) & 3u] * xh[j + 6];
            s3 += cb[(b1 >> 6) & 3u] * xh[j + 7];
        }
        return (s0 + s1) + (s2 + s3);
    }
    if (bits == 4 && (bit_off & 7u) == 0) {
        const uint8_t *q = p + (bit_off >> 3);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (j = 0; j < g; j += 4) {
            uint8_t b0 = q[0], b1 = q[1];
            q += 2;
            s0 += cb[b0 & 15u] * xh[j];
            s1 += cb[b0 >> 4]  * xh[j + 1];
            s2 += cb[b1 & 15u] * xh[j + 2];
            s3 += cb[b1 >> 4]  * xh[j + 3];
        }
        return (s0 + s1) + (s2 + s3);
    }

    /* Generic bit reader (covers bits == 3, and any unaligned start). Only
     * fetches the second byte when the field actually straddles, so the last
     * index of the last row never reads past the blob. */
    for (j = 0; j < g; j++) {
        uint32_t b   = bit_off + j * bits;
        uint32_t byt = b >> 3;
        uint32_t sh  = b & 7u;
        uint32_t w   = p[byt];
        if (sh + bits > 8)
            w |= (uint32_t)p[byt + 1] << 8;
        s += cb[(w >> sh) & ((1u << bits) - 1u)] * xh[j];
    }
    return s;
}

/* Row-range variant. `base` shifts the tensor row that output i maps to, so
 * the mHC phi tensors (all 27 layers stacked into one tensor) can hand out a
 * per-layer slice and still be split across cores. */
static ND_HOT void gemv_rows_offset(void *vc, uint32_t i0, uint32_t i1)
{
    const gemv_ctx *c = (const gemv_ctx *)vc;
    uint32_t        i;

    for (i = i0; i < i1; i++) {
        uint32_t        r   = c->base + i;
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * c->g * c->bits, c->bits, c->g, c->cb,
                             c->xh + (size_t)gi * c->g);
        c->y[i] = acc;
    }
}

ND_HOT void nd_cq_gemv_rows(const nd_cact *c, const nd_tensor *t, const void *blob,
                     const float *xh, uint32_t r0, uint32_t nrows, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    gemv_ctx       ctx;

    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.xh       = xh;
    ctx.cb       = nd_cact_codebook(c, t->bits);
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = t->group;
    ctx.bits     = t->bits;
    ctx.rowbytes = rowbytes;
    ctx.base     = r0;

    nd_parallel_rows(gemv_rows_offset, &ctx, nrows);
}



static ND_HOT void gemv_rows_generic(void *vc, uint32_t r0, uint32_t r1)
{
    const gemv_ctx *c = (const gemv_ctx *)vc;
    uint32_t        r;

    for (r = r0; r < r1; r++) {
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * c->g * c->bits, c->bits, c->g, c->cb,
                             c->xh + (size_t)gi * c->g);
        c->y[r] = acc;
    }
}

ND_HOT void nd_cq_gemv_prepared(const nd_cact *c, const nd_tensor *t, const void *blob,
                         const float *xh, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    gemv_ctx       ctx;

    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.xh       = xh;
    ctx.cb       = nd_cact_codebook(c, t->bits);
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = t->group;
    ctx.bits     = t->bits;
    ctx.rowbytes = rowbytes;

    nd_parallel_rows(gemv_rows_generic, &ctx, out);
}

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
static void pie_side_build(const float *cb, const float *xh, uint32_t in_pad);
#endif

ND_HOT void nd_cq_lut_build(const nd_cact *c, const float *xh, uint32_t in_pad,
                            float *lut)
{
    const float *cb = nd_cact_codebook(c, 2);
    uint32_t     p;

    /* T[i0 | (i1 << 2)] = cb[i0]*xh[2p] + cb[i1]*xh[2p+1], matching the
     * LSB-first packing (the low 2 bits of a nibble are the earlier weight). */
    for (p = 0; p < in_pad / 2; p++) {
        float  x0 = xh[2 * p];
        float  x1 = xh[2 * p + 1];
        float  a0 = cb[0] * x0, a1 = cb[1] * x0, a2 = cb[2] * x0, a3 = cb[3] * x0;
        float  b0 = cb[0] * x1, b1 = cb[1] * x1, b2 = cb[2] * x1, b3 = cb[3] * x1;
        float *T  = lut + (size_t)p * 16;

        T[0]  = a0 + b0; T[1]  = a1 + b0; T[2]  = a2 + b0; T[3]  = a3 + b0;
        T[4]  = a0 + b1; T[5]  = a1 + b1; T[6]  = a2 + b1; T[7]  = a3 + b1;
        T[8]  = a0 + b2; T[9]  = a1 + b2; T[10] = a2 + b2; T[11] = a3 + b2;
        T[12] = a0 + b3; T[13] = a1 + b3; T[14] = a2 + b3; T[15] = a3 + b3;
    }
#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
    pie_side_build(cb, xh, in_pad);
#endif
}

/* One group's contribution: 8 weights per iteration, four table lookups. */
static ND_HOT float dot_group_lut2(const uint8_t *q, const float *T, uint32_t g)
{
    float    s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint32_t j;

    for (j = 0; j < g; j += 8) {
        uint8_t b0 = q[0], b1 = q[1];
        q += 2;
        s0 += T[b0 & 15u];
        s1 += T[16 + (b0 >> 4)];
        s2 += T[32 + (b1 & 15u)];
        s3 += T[48 + (b1 >> 4)];
        T += 64;                      /* 4 pairs consumed */
    }
    return (s0 + s1) + (s2 + s3);
}

/* Default splitter: run everything on the calling thread. */
static void rows_serial(nd_row_fn fn, void *ctx, uint32_t nrows)
{
    fn(ctx, 0, nrows);
}

void (*nd_parallel_rows)(nd_row_fn fn, void *ctx, uint32_t nrows) = rows_serial;

typedef struct {
    const nd_tensor *t;
    const uint8_t   *packed;
    const uint16_t  *norms;
    const float     *lut;
    float           *y;
    uint32_t         ngroup, gbytes, gpairs, g, rowbytes;
} lut2_ctx;

static ND_HOT void lut2_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const lut2_ctx *c = (const lut2_ctx *)vc;
    uint32_t        r;

    for (r = r0; r < r1; r++) {
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group_lut2(row + (size_t)gi * c->gbytes,
                                  c->lut + (size_t)gi * c->gpairs * 16, c->g);
        c->y[r] = acc;
    }
}

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
/* ------------------------------------------------------------------------
 * Flash staging for the 2-bit GEMV weights.
 *
 * needle2's tensors are mmap'd from SPI flash: ~33 MB/s sustained through
 * the cache, which is exactly what the LUT kernel achieves per token - the
 * per-token projections are flash-bandwidth-bound, not compute-bound.
 * Stage each tensor's packed rows + norms into PSRAM on first use (they are
 * read-only afterwards). nd_parallel_rows runs us on both cores, so the
 * staging table is mutex-protected; copies happen during the boot prime.
 * ------------------------------------------------------------------------ */
#define ND_P4_STAGE_MAX    96
#define ND_P4_STAGE_BUDGET (10u * 1024u * 1024u + 512u * 1024u)  /* 10.5 MB */

typedef struct {
    const void *src;
    uint8_t    *dst;
    uint32_t    bytes;
} nd_p4_stage_ent;

static nd_p4_stage_ent  s_stage[ND_P4_STAGE_MAX];
static uint32_t         s_stage_n, s_stage_bytes;
static SemaphoreHandle_t s_stage_mu;

void nd_quant_p4_init(void)
{
    if (s_stage_mu == NULL)
        s_stage_mu = xSemaphoreCreateMutex();
}

static const void *nd_p4_stage(const void *src, uint32_t bytes)
{
    nd_p4_stage_ent *e;
    uint32_t         i;
    uint8_t         *dst = NULL;
    int64_t          t0;

    if (s_stage_mu == NULL || src == NULL || bytes == 0)
        return src;
    xSemaphoreTake(s_stage_mu, portMAX_DELAY);
    for (i = 0; i < s_stage_n; i++)
        if (s_stage[i].src == src) {
            xSemaphoreGive(s_stage_mu);
            return s_stage[i].dst;
        }
    if (s_stage_n < ND_P4_STAGE_MAX &&
        s_stage_bytes + bytes <= ND_P4_STAGE_BUDGET) {
        dst = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM |
                                                MALLOC_CAP_8BIT);
        if (dst != NULL) {
            t0 = esp_timer_get_time();
            memcpy(dst, src, bytes);   /* flash -> PSRAM, one time */
            e = &s_stage[s_stage_n++];
            e->src = src; e->dst = dst; e->bytes = bytes;
            s_stage_bytes += bytes;
            printf("EVT lut2 staged %u bytes=%u total=%u ms=%d\n",
                   (unsigned)s_stage_n, (unsigned)bytes,
                   (unsigned)s_stage_bytes,
                   (int)((esp_timer_get_time() - t0) / 1000));
            fflush(stdout);
        }
    }
    xSemaphoreGive(s_stage_mu);
    return dst != NULL ? dst : src;
}
/* ------------------------------------------------------------------------
 * Stage 2: PIE bit-plane kernel for the 2-bit GEMV.
 *
 * The float LUT kernel is a gather (4 table loads per 8 weights); PIE has
 * no gather, so decompose instead: w = cb[lo + 2*hi] expands bilinearly
 * and the three bit-plane sums come out of vector MACs:
 *   dot = sx * (cb0*Z + (cb1-cb0)*S_lo + (cb2-cb0)/2*S2_hi
 *                - (cb3-cb2-cb1+cb0)*S_lhi_neg)
 * with Z = sum(qi8) per group, S_lo/S2_hi/S_lhi from int8 x (qi8) and the
 * packed bits via uniform byte shifts. Lane order of the permuted x is
 * chosen so every shift is uniform (2 bits per weight, stride 4).
 * ------------------------------------------------------------------------ */
static int      s_lut2_mode;      /* 0 float, 1 pie, 2 armed-verify */
static int      s_side_ok;
static int      s_side_pad;       /* in_pad the side tables were built for */
static uint8_t *s_px;             /* ng*128: permuted int8 x */
static float   *s_ZC;             /* ng*4 floats: ZC, A, B, C */
static int32_t *s_Zg;             /* ng: per-group sum(qi8) */

static const uint8_t s_pie_c12[32] __attribute__((aligned(16))) = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};

/* Build the per-token side tables for the PIE bit-plane kernel: the int8
 * quantisation of xh (permuted into the lane order the asm extracts), the
 * per-group integer sums Z, and the float bilinear coefficients. */
static void pie_side_build(const float *cb, const float *xh, uint32_t in_pad)
{
    uint32_t ng = in_pad / 128u, gi;

    if (in_pad % 128u || ng == 0)
        return;
    if (s_px == NULL || s_side_pad != (int)in_pad) {
        if (s_px) {
            heap_caps_free(s_px); heap_caps_free(s_ZC); heap_caps_free(s_Zg);
        }
        s_px  = heap_caps_aligned_alloc(16, ng * 128u, MALLOC_CAP_INTERNAL |
                                                       MALLOC_CAP_8BIT);
        s_ZC  = heap_caps_aligned_alloc(16, ng * 4 * sizeof(float),
                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        s_Zg  = heap_caps_aligned_alloc(16, ng * sizeof(int32_t),
                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (s_px == NULL || s_ZC == NULL || s_Zg == NULL) {
            if (s_px)  { heap_caps_free(s_px);  s_px  = NULL; }
            if (s_ZC)  { heap_caps_free(s_ZC);  s_ZC  = NULL; }
            if (s_Zg)  { heap_caps_free(s_Zg);  s_Zg  = NULL; }
            s_side_ok = 0;
            return;
        }
        s_side_pad = (int)in_pad;
    }
    for (gi = 0; gi < ng; gi++) {
        const float *xg = xh + gi * 128u;
        float mx = 0.0f, sx, *co = s_ZC + gi * 4;
        int32_t Z = 0;
        uint32_t j;
        int8_t  q[128];

        for (j = 0; j < 128u; j++) {
            float a = xg[j] < 0 ? -xg[j] : xg[j];
            if (a > mx) mx = a;
        }
        sx = mx > 0.0f ? mx / 127.0f : 1.0f;
        for (j = 0; j < 128u; j++) {
            int qi = (int)(xg[j] / sx + (xg[j] >= 0 ? 0.5f : -0.5f));
            if (qi > 127)  qi = 127;
            if (qi < -127) qi = -127;
            q[j] = (int8_t)qi;
            Z   += qi;
        }
        s_Zg[gi] = Z;
        /* lane order: px[gi][h*4+k][b] = q[h*64 + b*4 + k] - matches the
         * uniform byte shifts (2k) the asm applies per half. */
        {
            uint8_t *px = s_px + gi * 128u;
            uint32_t h, k, b;
            for (h = 0; h < 2u; h++)
                for (k = 0; k < 4u; k++)
                    for (b = 0; b < 16u; b++)
                        px[(h * 4u + k) * 16u + b] =
                            (uint8_t)q[h * 64u + b * 4u + k];
        }
        /* w = cb0 + (cb1-cb0)lo + (cb2-cb0)hi + (cb3-cb2-cb1+cb0)lo*hi,
         * folded with the x scale; S_hi arrives doubled ({0,2} lanes) and
         * S_lhi negated (0xFF = -1 compare masks). */
        co[0] = sx * cb[0] * (float)Z;
        co[1] = sx * (cb[1] - cb[0]);
        co[2] = sx * (cb[2] - cb[0]) * 0.5f;
        co[3] = -sx * (cb[3] - cb[2] - cb[1] + cb[0]);
    }
    s_side_ok = 1;
}

/* One 128-weight group -> out[0]=S_lo, out[1]=2*S_hi, out[2]=-S_lhi
 * (each a 32-bit xacc read whose value fits in int16). */
static ND_HOT void pie_lut2_group(const uint8_t *w, const uint8_t *px,
                                  int32_t *out3)
{
    asm volatile(
        ".option push\n\t"
        ".option arch, +xesppie, +xespv\n\t"
        "mv   x30, %[w]\n\t"
        "mv   x28, %[c]\n\t"
        "mv   x29, %[px]\n\t"
        "esp.vld.128.ip q2, x28, 16\n\t"        /* q2 = 0x01 x16 */
        "esp.vld.128.ip q3, x28, 16\n\t"        /* q3 = 0x02 x16 */
        "esp.vadd.u8     q4, q2, q3\n\t"        /* q4 = 0x03 x16 */
        "esp.vld.128.ip  q0, x30, 16\n\t"       /* weights 0..63  */
        "esp.vld.128.ip  q1, x30, 16\n\t"       /* weights 64..127*/

        "esp.zero.xacc\n\t"
        "mv   x31, x29\n\t"
        "esp.orq  q5, q0, q0\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.orq  q5, q1, q1\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q2\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.movx.r.xacc.l x30\n\t"
        "mv   t0, x30\n\t"                      /* t0 = S_lo */

        "esp.zero.xacc\n\t"
        "mv   x31, x29\n\t"
        "esp.orq  q5, q0, q0\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.orq  q5, q1, q1\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q3\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.movx.r.xacc.l x30\n\t"
        "mv   t1, x30\n\t"                      /* t1 = 2*S_hi */

        "esp.zero.xacc\n\t"
        "mv   x31, x29\n\t"
        "esp.orq  q5, q0, q0\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.orq  q5, q1, q1\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.vsrd.8 q5, q5, q3\n\t"
        "esp.andq q6, q5, q4\n\t"
        "esp.vcmp.eq.s8 q6, q6, q4\n\t"
        "esp.vld.128.ip q7, x31, 16\n\t"
        "esp.vmulas.s8.xacc q6, q7\n\t"
        "esp.movx.r.xacc.l x30\n\t"
        "mv   t2, x30\n\t"                      /* t2 = -S_lhi */

        "sw   t0, 0(%[o])\n\t"
        "sw   t1, 4(%[o])\n\t"
        "sw   t2, 8(%[o])\n\t"
        ".option pop\n\t"
        :
        : [w] "r"(w), [px] "r"(px), [c] "r"(s_pie_c12), [o] "r"(out3)
        : "t0", "t1", "t2", "x28", "x29", "x30", "x31", "memory");
}

static ND_HOT void pie_lut2_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const lut2_ctx *c = (const lut2_ctx *)vc;
    uint32_t        r;

    for (r = r0; r < r1; r++) {
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++) {
            int32_t s[3];
            int32_t slo, shi, slh;
            const float *co = s_ZC + gi * 4;

            pie_lut2_group(row + (size_t)gi * 32u,
                           s_px + (size_t)gi * 128u, s);
            slo = (int16_t)s[0]; shi = (int16_t)s[1]; slh = (int16_t)s[2];

            acc += nd_f16(nrm[gi]) *
                   (co[0] + co[1] * (float)slo + co[2] * (float)shi +
                    co[3] * (float)slh);
        }
        c->y[r] = acc;
    }
}

/* First-use verification: 4 rows through both kernels, PIE must track the
 * float LUT path within the int8-quantisation error. */
static void pie_lut2_verify(const lut2_ctx *c)
{
    float yf[4], yp[4];
    float mean = 0.0f, worst = 0.0f;
    uint32_t i;
    int ok = 1;

    xSemaphoreTake(s_stage_mu, portMAX_DELAY);
    if (!s_side_ok || s_lut2_mode != 2) {
        xSemaphoreGive(s_stage_mu);
        return;
    }
    {
        float *ysave = c->y;
        lut2_ctx cf = *c, cp = *c;
        cf.y = yf; cp.y = yp;
        lut2_rows(&cf, 0, 4);
        pie_lut2_rows(&cp, 0, 4);
        (void)ysave;
    }
    for (i = 0; i < 4u; i++) mean += yf[i] < 0 ? -yf[i] : yf[i];
    mean /= 4.0f;
    for (i = 0; i < 4u; i++) {
        float d = yp[i] - yf[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    ok = worst <= 0.10f * (mean + 1e-6f);   /* int8 per-group-max quant noise
                                              * is ~2-3% on real activations
                                              * (heavy-tailed x); structural
                                              * kernel bugs land >50% */
    if (mean < 1e-5f) {
        xSemaphoreGive(s_stage_mu);   /* degenerate input: retry next gemv */
        return;
    }
    s_lut2_mode = ok ? 1 : 0;
    printf("EVT lut2 pie %s worst=%.5f mean=%.5f\n",
           ok ? "ok" : "mismatch", (double)worst, (double)mean);
    fflush(stdout);
    xSemaphoreGive(s_stage_mu);
}

/* Synthetic on-device check: random weights + x through both kernels.
 * Reports the three raw asm sums per group against scalar references and
 * the full-group dot against the float LUT - pinpoints shift/mask/coef
 * bugs. Returns 0 ok. */
int nd_quant_p4_lut2_check(void)
{
    static const float cb[4] = { -0.45f, -0.12f, 0.31f, 0.78f };
    static float    xh[512], lut[512 / 2 * 16];
    static uint8_t  w[128];
    int32_t  s[3];
    uint32_t gi, j, p, seed = 12345u, bad = 0;
    float    row_pie = 0.0f, row_flt = 0.0f;

    for (j = 0; j < 512u; j++) {
        seed = seed * 1664525u + 1013904223u;
        xh[j] = ((float)(int32_t)(seed >> 8) / 8388608.0f) - 1.0f;
    }
    for (j = 0; j < 128u; j++) {
        seed = seed * 1664525u + 1013904223u;
        w[j] = (uint8_t)(seed >> 13);
    }
    pie_side_build(cb, xh, 512u);

    /* float LUT exactly as nd_cq_lut_build */
    for (p = 0; p < 256u; p++) {
        float x0 = xh[2 * p], x1 = xh[2 * p + 1];
        float a[4] = { cb[0] * x0, cb[1] * x0, cb[2] * x0, cb[3] * x0 };
        float b[4] = { cb[0] * x1, cb[1] * x1, cb[2] * x1, cb[3] * x1 };
        float *T  = lut + p * 16u;
        uint32_t i0, i1;
        for (i1 = 0; i1 < 4u; i1++)
            for (i0 = 0; i0 < 4u; i0++)
                T[i0 | (i1 << 2)] = a[i0] + b[i1];
    }

    for (gi = 0; gi < 4u; gi++) {
        const uint8_t *qw = w + gi * 32u;
        int32_t rlo = 0, rhi2 = 0, rlneg = 0;
        float   fflt;

        pie_lut2_group(qw, s_px + gi * 128u, s);
        for (j = 0; j < 128u; j++) {
            uint32_t v  = (qw[j >> 2] >> (2u * (j & 3u))) & 3u;
            int32_t  qx = (int8_t)s_px[gi * 128u +
                                       ((j / 64u) * 4u + (j & 3u)) * 16u +
                                       ((j % 64u) / 4u)];
            rlo   += (v & 1u)      * qx;
            rhi2  += ((v >> 1) & 1u) * 2 * qx;
            rlneg -= (v == 3u)     * qx;
        }
        if (s[0] != rlo || s[1] != rhi2 || s[2] != rlneg) {
            printf("EVT lut2 dbg gi=%u lo %d/%d hi2 %d/%d lneg %d/%d\n",
                   (unsigned)gi, (int)s[0], (int)rlo, (int)s[1], (int)rhi2,
                   (int)s[2], (int)rlneg);
            bad++;
        }
        fflt = dot_group_lut2(qw, lut + gi * 64u * 16u, 128u);
        row_flt += fflt;
        row_pie += (s_ZC[gi * 4u + 0u] + s_ZC[gi * 4u + 1u] * (int16_t)s[0] +
                    s_ZC[gi * 4u + 2u] * (int16_t)s[1] +
                    s_ZC[gi * 4u + 3u] * (int16_t)s[2]);
    }
    printf("EVT lut2 dbg sums=%s flt=%.4f pie=%.4f rel=%.5f\n",
           bad ? "BAD" : "ok", (double)row_flt, (double)row_pie,
           (double)((row_pie - row_flt) / (row_flt + 1e-9f)));
    fflush(stdout);
    return bad || (row_pie - row_flt) / (row_flt + 1e-9f) > 0.02f ||
                   (row_pie - row_flt) / (row_flt + 1e-9f) < -0.02f;
}

void nd_quant_p4_lut2_enable(void)
{
    if (nd_quant_p4_lut2_check() == 0)
        s_lut2_mode = 2;    /* synthetic ok; verify on first real gemv too */
    else
        printf("EVT lut2 pie disabled (synthetic check failed)\n"),
            fflush(stdout);
}
#endif /* ESP32P4 PIE lut2 */

ND_HOT void nd_cq_gemv_lut2(const nd_tensor *t, const void *blob,
                            const float *lut, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       g        = t->group;
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    lut2_ctx       ctx;

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
    packed = (const uint8_t *)nd_p4_stage(
        packed, (uint32_t)((size_t)out * rowbytes +
                           (size_t)out * nd_cq_groups(t) * 2));
#endif

    ctx.t        = t;
    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.lut      = lut;
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = g;
    ctx.gbytes   = g / 4;              /* 2 bits per weight */
    ctx.gpairs   = g / 2;
    ctx.rowbytes = rowbytes;

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
    if (s_lut2_mode == 2)
        pie_lut2_verify(&ctx);
    if (s_lut2_mode == 1 && g == 128u && s_side_ok &&
        s_side_pad == (int)(ctx.ngroup * 128u))
        nd_parallel_rows(pie_lut2_rows, &ctx, out);
    else
        nd_parallel_rows(lut2_rows, &ctx, out);
#else
    nd_parallel_rows(lut2_rows, &ctx, out);
#endif
}

ND_HOT void nd_cq_gemv_gather(const nd_cact *c, const nd_tensor *t, const void *blob,
                              const float *xh, const uint32_t *ids, uint32_t n,
                              float *y)
{
    const uint8_t  *packed   = (const uint8_t *)blob;
    uint32_t        out      = t->shape[0];
    uint32_t        g        = t->group;
    uint32_t        ngroup   = nd_cq_groups(t);
    uint32_t        rowbytes = nd_cq_row_bytes(t);
    const uint16_t *norms    = (const uint16_t *)(const void *)(packed +
                                (size_t)out * rowbytes);
    const float    *cb       = nd_cact_codebook(c, t->bits);
    uint32_t        i;

    for (i = 0; i < n; i++) {
        const uint8_t  *row = packed + (size_t)ids[i] * rowbytes;
        const uint16_t *nrm = norms + (size_t)ids[i] * ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * g * t->bits, t->bits, g, cb,
                             xh + (size_t)gi * g);
        y[i] = acc;
    }
}

void nd_cq_gemv(const nd_cact *c, const nd_tensor *t, const void *blob,
                const float *x, float *scratch, float *y)
{
    nd_cq_prepare(t, x, scratch);
    nd_cq_gemv_prepared(c, t, blob, scratch, y);
}

typedef struct {
    const nd_cact   *c;
    const nd_tensor *t;
    const uint8_t   *packed;
    const uint16_t  *norms;
    const float     *xh;
    float           *y;
    uint32_t         T;
    uint32_t         r0;        /* first tensor row; Y column = row - r0 */
    uint32_t         ystride;
    float           *wbuf[2];   /* one decoded row per core (nd_parallel_rows
                                 * hands core 1 the rows from nrows/2) */
    const float     *tab2;      /* 2-bit: byte -> its 4 codebook values */
} gemm_ctx;

/* One output row at a time: decode the whole row once (2-bit through a
 * byte -> 4 weights table), then dot it with every activation row using four
 * independent accumulators. On the LX7 a single running sum serialises on
 * the FPU's multiply-add latency; the first version of this loop ran at ~9
 * cycles per multiply-add on the board. */
#define GEMM_MAX_IN 2048

static ND_HOT void gemm_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const gemm_ctx  *g      = (const gemm_ctx *)vc;
    const nd_tensor *t      = g->t;
    uint32_t         out    = g->ystride;
    uint32_t         gs     = t->group;
    uint32_t         in_pad = nd_cq_in_pad(t);
    uint32_t         ngroup = in_pad / gs;
    uint32_t         rowb   = nd_cq_row_bytes(t);
    uint32_t         bits   = t->bits;
    uint32_t         mask   = (1u << bits) - 1u;
    const float     *cb     = nd_cact_codebook(g->c, bits);
    float           *w      = g->wbuf[r0 == 0 ? 0 : 1];
    uint32_t         r, gi, j, k;

    for (r = r0; r < r1; r++) {
        const uint8_t  *p   = g->packed + (size_t)(r + g->r0) * rowb;
        const uint16_t *nrm = g->norms + (size_t)(r + g->r0) * ngroup;

        /* In the Hadamard domain: the activation was transformed by
         * nd_cq_prepare, so the weights need no inverse transform. */
        for (gi = 0; gi < ngroup; gi++) {
            float    norm = nd_f16(nrm[gi]);
            float   *wg   = w + (size_t)gi * gs;
            if (bits == 2) {
                const uint8_t *q = p + gi * gs / 4;
                for (j = 0; j < gs / 4; j++) {
                    const float *e = g->tab2 + 4u * q[j];
                    wg[4 * j]     = e[0] * norm;
                    wg[4 * j + 1] = e[1] * norm;
                    wg[4 * j + 2] = e[2] * norm;
                    wg[4 * j + 3] = e[3] * norm;
                }
            } else {
                uint32_t bit_off = gi * gs * bits;
                for (j = 0; j < gs; j++) {
                    uint32_t b = bit_off + j * bits;
                    uint32_t v = p[b >> 3];
                    if ((b & 7u) + bits > 8)
                        v |= (uint32_t)p[(b >> 3) + 1] << 8;
                    wg[j] = cb[(v >> (b & 7u)) & mask] * norm;
                }
            }
        }
        /* Two activation rows per pass: each weight is loaded once for two
         * multiply-adds. The LX7 issues one instruction a cycle, so loads,
         * not arithmetic, set the pace. */
        for (k = 0; k + 1 < g->T; k += 2) {
            const float *x = g->xh + (size_t)k * in_pad, *z = x + in_pad;
            float        a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float        b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            for (j = 0; j < in_pad; j += 4) {
                float w0 = w[j], w1 = w[j + 1], w2 = w[j + 2], w3 = w[j + 3];
                a0 += w0 * x[j];     b0 += w0 * z[j];
                a1 += w1 * x[j + 1]; b1 += w1 * z[j + 1];
                a2 += w2 * x[j + 2]; b2 += w2 * z[j + 2];
                a3 += w3 * x[j + 3]; b3 += w3 * z[j + 3];
            }
            g->y[(size_t)k * out + r]       = (a0 + a1) + (a2 + a3);
            g->y[(size_t)(k + 1) * out + r] = (b0 + b1) + (b2 + b3);
        }
        for (; k < g->T; k++) {
            const float *x = g->xh + (size_t)k * in_pad;
            float        a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (j = 0; j < in_pad; j += 4) {
                a0 += w[j] * x[j];
                a1 += w[j + 1] * x[j + 1];
                a2 += w[j + 2] * x[j + 2];
                a3 += w[j + 3] * x[j + 3];
            }
            g->y[(size_t)k * out + r] = (a0 + a1) + (a2 + a3);
        }
    }
}

void nd_cq_gemm_prepared(const nd_cact *c, const nd_tensor *t, const void *blob,
                         const float *xh, uint32_t T, float *y)
{
    nd_cq_gemm_rows(c, t, blob, xh, T, 0, t->shape[0], y);
}

/* ------------------------------------------------------------ int8 path */

int nd_cq_int8 = 1;

/* sum(a[i] * b[i]) over 128 int8s; both 16-byte aligned. */
static ND_HOT int32_t dot128_s8(const int8_t *a, const int8_t *b)
{
#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
    /* ESP32-P4 HP core: ESP PIE DSP extension (xesppie + xespv vector file).
     * Same algorithm as the Xtensa branch below: 8 x 16-lane signed MACs
     * accumulated into the 32-bit xacc register. The fused
     * esp.vmulas.s8.xacc.ld.ip form performs the MAC, the next a-tile load
     * and the pointer bump in one instruction; a/b streams are double
     * buffered across q0/q2 and q1/q3 so no load is consumed by the
     * instruction right after it (esp-nn notes a 3-instruction latency).
     * Integer arithmetic only: the result is bit-identical to the C loop.
     * The .option arch directives let this compile with the toolchain's
     * default -march (no project-wide flag change needed). */
    int32_t r;
    __asm__ volatile(
        ".option push                                    \n\t"
        ".option arch, +xesppie, +xespv                  \n\t"
        "mv     x30, %1                                  \n\t"
        "mv     x31, %2                                  \n\t"
        "esp.zero.xacc                                   \n\t"
        "esp.vld.128.ip q0, x30, 16   \n\t"   /* q0 = a[0:16)   */
        "esp.vld.128.ip q2, x30, 16   \n\t"   /* q2 = a[16:32)  */
        "esp.vld.128.ip q1, x31, 16   \n\t"   /* q1 = b[0:16)   */
        "esp.vld.128.ip q3, x31, 16   \n\t"   /* q3 = b[16:32)  */
        "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t" /* += a0*b0;  q0=a[32:48)   */
        "esp.vld.128.ip q1, x31, 16   \n\t"   /* q1 = b[32:48)  */
        "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t" /* += a16*b16; q2=a[48:64)  */
        "esp.vld.128.ip q3, x31, 16   \n\t"   /* q3 = b[48:64)  */
        "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t" /* += a32*b32; q0=a[64:80)  */
        "esp.vld.128.ip q1, x31, 16   \n\t"   /* q1 = b[64:80)  */
        "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t" /* += a48*b48; q2=a[80:96)  */
        "esp.vld.128.ip q3, x31, 16   \n\t"   /* q3 = b[80:96)  */
        "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t" /* += a64*b64; q0=a[96:112) */
        "esp.vld.128.ip q1, x31, 16   \n\t"   /* q1 = b[96:112) */
        "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t" /* += a80*b80; q2=a[112:128)*/
        "esp.vld.128.ip q3, x31, 16   \n\t"   /* q3 = b[112:128)*/
        "esp.vmulas.s8.xacc q0, q1    \n\t"   /* += a[96:112)*b[96:112)     */
        "esp.vmulas.s8.xacc q2, q3    \n\t"   /* += a[112:128)*b[112:128)   */
        "esp.movx.r.xacc.l x30        \n\t"   /* x30 = xacc (low 32 bits)   */
        "mv     %0, x30               \n\t"
        ".option pop                                     \n\t"
        : "=r"(r)
        : "r"(a), "r"(b)
        : "x30", "x31", "memory");
    return r;
#elif defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3)
    int32_t r;
    __asm__ volatile(
        "ee.zero.accx\n"
        "ee.vld.128.ip q0, %1, 16\n"  "ee.vld.128.ip q1, %2, 16\n"
        "ee.vld.128.ip q2, %1, 16\n"  "ee.vld.128.ip q3, %2, 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "ee.vld.128.ip q4, %1, 16\n"  "ee.vld.128.ip q5, %2, 16\n"
        "ee.vmulas.s8.accx q2, q3\n"
        "ee.vld.128.ip q6, %1, 16\n"  "ee.vld.128.ip q7, %2, 16\n"
        "ee.vmulas.s8.accx q4, q5\n"
        "ee.vld.128.ip q0, %1, 16\n"  "ee.vld.128.ip q1, %2, 16\n"
        "ee.vmulas.s8.accx q6, q7\n"
        "ee.vld.128.ip q2, %1, 16\n"  "ee.vld.128.ip q3, %2, 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "ee.vld.128.ip q4, %1, 16\n"  "ee.vld.128.ip q5, %2, 16\n"
        "ee.vmulas.s8.accx q2, q3\n"
        "ee.vld.128.ip q6, %1, 16\n"  "ee.vld.128.ip q7, %2, 16\n"
        "ee.vmulas.s8.accx q4, q5\n"
        "ee.vmulas.s8.accx q6, q7\n"
        "rur.accx_0 %0\n"           /* |sum| <= 128*127*127: fits 32 bits */
        : "=r"(r), "+r"(a), "+r"(b) :: "memory");
    return r;
#else
    int32_t  r = 0;
    uint32_t i;
    for (i = 0; i < 128; i++)
        r += (int32_t)a[i] * b[i];
    return r;
#endif
}

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
/* Boot-time check of the PIE dot128_s8 above against the portable C loop
 * on deterministic vectors (LCG, plus the extreme values +127/-128), with
 * a crude timing race over NITER calls. Integer accumulation is exact, so
 * any mismatch means the kernel or the silicon is misbehaving.
 * Returns 0 when every vector matches (and fills *pie_us / *c_us with the
 * elapsed microseconds of each timing loop), or the 1-based number of the
 * first failing vector. */
int nd_quant_p4_check(int64_t *pie_us, int64_t *c_us)
{
    enum { NT = 10, NITER = 5000 };
    static int8_t a[NT][128] __attribute__((aligned(16)));
    static int8_t b[NT][128] __attribute__((aligned(16)));
    uint32_t seed = 0x12345678u;
    volatile int32_t sink = 0;
    int32_t acc;
    int64_t t0;
    int n, i;

    for (n = 0; n < NT; n++) {
        for (i = 0; i < 128; i++) {
            seed = seed * 1664525u + 1013904223u;
            a[n][i] = (int8_t)(seed >> 24);
            seed = seed * 1664525u + 1013904223u;
            b[n][i] = (int8_t)(seed >> 25);
        }
    }
    for (i = 0; i < 128; i++) {           /* extremes: +127 / -128 */
        a[NT - 2][i] = (int8_t)0x7F;
        b[NT - 2][i] = (int8_t)0x80;
        a[NT - 1][i] = (int8_t)0x80;
        b[NT - 1][i] = (int8_t)0x7F;
    }

    for (n = 0; n < NT; n++) {
        int32_t pie = dot128_s8(a[n], b[n]);
        int32_t ref = 0;
        for (i = 0; i < 128; i++)
            ref += (int32_t)a[n][i] * b[n][i];
        if (pie != ref)
            return n + 1;
    }

    t0 = esp_timer_get_time();
    acc = 0;
    for (n = 0; n < NITER; n++)
        acc += dot128_s8(a[n & (NT - 2)], b[n & (NT - 3)]);
    sink = acc;
    *pie_us = esp_timer_get_time() - t0;

    t0 = esp_timer_get_time();
    acc = 0;
    for (n = 0; n < NITER; n++) {
        const int8_t *pa = a[n & (NT - 2)], *pb = b[n & (NT - 3)];
        int32_t r = 0;
        for (i = 0; i < 128; i++)
            r += (int32_t)pa[i] * pb[i];
        acc += r;
    }
    sink = acc;
    *c_us = esp_timer_get_time() - t0;

    return (sink == 0x7FFFFFFF) ? -1 : 0;  /* keep sink live; never true */
}
#endif

typedef struct {
    const nd_tensor *t;
    const uint8_t   *packed;
    const uint16_t  *norms;
    const int8_t    *xq;        /* [T][in_pad] activation tile */
    const float     *sx;        /* [T][ngroup] its per-group scales */
    float           *y;
    uint32_t         T, r0, ystride;
    int8_t          *wq[2];     /* one decoded row per core */
    const uint32_t  *tab2;      /* 2-bit: byte -> 4 int8 levels */
    const uint16_t  *tab4;      /* 4-bit: byte -> 2 int8 levels */
    float            inv_sw;    /* int8 level -> codebook value */
} gemm8_ctx;

static ND_HOT void gemm8_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const gemm8_ctx *g      = (const gemm8_ctx *)vc;
    const nd_tensor *t      = g->t;
    uint32_t         in_pad = nd_cq_in_pad(t);
    uint32_t         ngroup = in_pad / 128;
    uint32_t         rowb   = nd_cq_row_bytes(t);
    int8_t          *wq     = g->wq[r0 == 0 ? 0 : 1];
    float            nrm[16];
    uint32_t         r, j, k, gi;

    for (r = r0; r < r1; r++) {
        const uint8_t  *p  = g->packed + (size_t)(r + g->r0) * rowb;
        const uint16_t *nm = g->norms + (size_t)(r + g->r0) * ngroup;
        if (t->bits == 2)
            for (j = 0; j < rowb; j++)
                ((uint32_t *)(void *)wq)[j] = g->tab2[p[j]];
        else
            for (j = 0; j < rowb; j++)
                ((uint16_t *)(void *)wq)[j] = g->tab4[p[j]];
        for (gi = 0; gi < ngroup; gi++)
            nrm[gi] = nd_f16(nm[gi]) * g->inv_sw;
        for (k = 0; k < g->T; k++) {
            const int8_t *x  = g->xq + (size_t)k * in_pad;
            const float  *sx = g->sx + (size_t)k * ngroup;
            float         acc = 0.0f;
            for (gi = 0; gi < ngroup; gi++)
                acc += nrm[gi] * sx[gi] * (float)dot128_s8(wq + gi * 128, x + gi * 128);
            g->y[(size_t)k * g->ystride + r] = acc;
        }
    }
}

static void *align16(void *p) { return (void *)(((uintptr_t)p + 15) & ~(uintptr_t)15); }

int nd_cq_q8_ok(const nd_tensor *t)
{
    return nd_cq_int8 && t->group == 128 && (t->bits == 2 || t->bits == 4) &&
           nd_cq_in_pad(t) <= 2048;
}

/* Round half away from zero with the FPU's float->int conversion (trunc.s):
 * newlib's lrintf is a software routine on the LX7. */
static inline int8_t q8(float v)
{
    return (int8_t)(int)(v + (v >= 0.0f ? 0.5f : -0.5f));
}

int nd_cq_q8_prepare(const nd_tensor *t, const float *x, uint32_t T, nd_q8_act *a)
{
    uint32_t in = t->shape[1], in_pad = nd_cq_in_pad(t), ng = in_pad / 128;
    size_t   bytes = (size_t)T * in_pad + (size_t)T * ng * sizeof(float) + 32;
    uint32_t k, g, j;
    float    blk[128];

    if (!nd_cq_q8_ok(t))
        return -1;
    /* Internal SRAM when it fits: every weight row reads every activation. */
    a->mem = bytes <= 64 * 1024 ? ND_QUANT_ALLOC(bytes) : NULL;
    if (!a->mem) {
#ifdef ESP_PLATFORM
        a->mem = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
#else
        a->mem = malloc(bytes);
#endif
    }
    if (!a->mem)
        return -1;
    a->xq = (int8_t *)align16(a->mem);
    a->sx = (float *)align16(a->xq + (size_t)T * in_pad);
    a->T = T;
    a->in_pad = in_pad;
    for (k = 0; k < T; k++)
        for (g = 0; g < ng; g++) {
            int8_t *q = a->xq + (size_t)k * in_pad + g * 128;
            float   m = 0.0f, s, inv;
            for (j = 0; j < 128; j++) {
                uint32_t col = g * 128 + j;
                blk[j] = col < in ? x[(size_t)k * in + col] : 0.0f;
            }
            nd_fwht(blk, 128);
            for (j = 0; j < 128; j++) {
                blk[j] *= 0.08838834764831845f;       /* 1/sqrt(128) */
                if (fabsf(blk[j]) > m)
                    m = fabsf(blk[j]);
            }
            s   = m > 0.0f ? m / 127.0f : 1.0f;
            inv = 1.0f / s;
            for (j = 0; j < 128; j++)
                q[j] = q8(blk[j] * inv);
            a->sx[(size_t)k * ng + g] = s;
        }
    return 0;
}

void nd_cq_q8_free(nd_q8_act *a)
{
    if (a->mem)
        ND_QUANT_FREE(a->mem);   /* heap_caps_free / free handle either heap */
    a->mem = NULL;
}

void nd_cq_q8_gemm(const nd_cact *c, const nd_tensor *t, const void *blob,
                   const nd_q8_act *a, uint32_t r0, uint32_t nrows, float *y)
{
    uint32_t     in_pad = a->in_pad, i;
    const float *cb = nd_cact_codebook(c, t->bits);
    uint32_t     nl = 1u << t->bits;
    float        mx = 0.0f, sw;
    int8_t       lv[16];
    gemm8_ctx    g;
    uint8_t     *buf;
    uint32_t    *tab2;
    uint16_t    *tab4;

    for (i = 0; i < nl; i++)
        if (fabsf(cb[i]) > mx)
            mx = fabsf(cb[i]);
    sw = 127.0f / mx;
    for (i = 0; i < nl; i++)
        lv[i] = q8(cb[i] * sw);
    buf  = (uint8_t *)ND_QUANT_ALLOC(32 + 2 * in_pad + 1024 + 512);
    g.wq[0] = (int8_t *)align16(buf);
    g.wq[1] = g.wq[0] + in_pad;
    tab2 = (uint32_t *)(void *)(g.wq[1] + in_pad);
    tab4 = (uint16_t *)(void *)(tab2 + 256);
    for (i = 0; i < 256; i++) {               /* LSB-first index packing */
        if (t->bits == 2)
            tab2[i] = (uint32_t)(uint8_t)lv[i & 3] | (uint32_t)(uint8_t)lv[(i >> 2) & 3] << 8 |
                      (uint32_t)(uint8_t)lv[(i >> 4) & 3] << 16 | (uint32_t)(uint8_t)lv[(i >> 6) & 3] << 24;
        else
            tab4[i] = (uint16_t)((uint8_t)lv[i & 15] | (uint16_t)(uint8_t)lv[i >> 4] << 8);
    }
    g.t = t;
    g.packed = (const uint8_t *)blob;
    g.norms = (const uint16_t *)(const void *)(g.packed + (size_t)t->shape[0] * nd_cq_row_bytes(t));
    g.r0 = r0;
    g.ystride = nrows;
    g.tab2 = tab2;
    g.tab4 = tab4;
    g.inv_sw = 1.0f / sw;
    if ((size_t)a->T * in_pad <= 32 * 1024) {
        g.xq = a->xq;
        g.sx = a->sx;
        g.y  = y;
        g.T  = a->T;
        nd_parallel_rows(gemm8_rows, &g, nrows);
    } else {
        /* A big batch lives in PSRAM, and every weight row reads all of
         * it: the stem's pointwise layers (1000 positions x 128) re-read
         * 128 KB per output row. Run all rows on 32 KB tiles in SRAM. */
        uint32_t ng = in_pad / 128, tile = (32 * 1024) / in_pad, k0;
        uint8_t *tb = (uint8_t *)ND_QUANT_ALLOC((size_t)tile * (in_pad + ng * sizeof(float)) + 32);
        int8_t  *xq = (int8_t *)align16(tb);
        float   *sx = (float *)align16(xq + (size_t)tile * in_pad);
        for (k0 = 0; k0 < a->T; k0 += tile) {
            uint32_t n = a->T - k0 < tile ? a->T - k0 : tile;
            memcpy(xq, a->xq + (size_t)k0 * in_pad, (size_t)n * in_pad);
            memcpy(sx, a->sx + (size_t)k0 * ng, (size_t)n * ng * sizeof(float));
            g.xq = xq;
            g.sx = sx;
            g.y  = y + (size_t)k0 * nrows;
            g.T  = n;
            nd_parallel_rows(gemm8_rows, &g, nrows);
        }
        ND_QUANT_FREE(tb);
    }
    ND_QUANT_FREE(buf);
}

static void gemm8(const nd_cact *c, const nd_tensor *t, const void *blob,
                  const float *xh, uint32_t T, uint32_t r0, uint32_t nrows, float *y)
{
    /* xh is already in the Hadamard domain here (nd_cq_gemm_* callers), so
     * quantise it directly. */
    uint32_t  in_pad = nd_cq_in_pad(t), ng = in_pad / 128, k, gi, j;
    nd_q8_act a;
    size_t    bytes = (size_t)T * in_pad + (size_t)T * ng * sizeof(float) + 32;

    a.mem = ND_QUANT_ALLOC(bytes);
    if (!a.mem) {
#ifdef ESP_PLATFORM
        a.mem = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
#else
        a.mem = malloc(bytes);
#endif
    }
    a.xq = (int8_t *)align16(a.mem);
    a.sx = (float *)align16(a.xq + (size_t)T * in_pad);
    a.T = T;
    a.in_pad = in_pad;
    for (k = 0; k < T; k++)
        for (gi = 0; gi < ng; gi++) {
            const float *x = xh + (size_t)k * in_pad + gi * 128;
            int8_t      *q = a.xq + (size_t)k * in_pad + gi * 128;
            float        m = 0.0f, s, inv;
            for (j = 0; j < 128; j++)
                if (fabsf(x[j]) > m)
                    m = fabsf(x[j]);
            s = m > 0.0f ? m / 127.0f : 1.0f;
            inv = 1.0f / s;
            for (j = 0; j < 128; j++)
                q[j] = q8(x[j] * inv);
            a.sx[(size_t)k * ng + gi] = s;
        }
    nd_cq_q8_gemm(c, t, blob, &a, r0, nrows, y);
    nd_cq_q8_free(&a);
}

/* Activation rows go through internal SRAM in tiles of at most this many
 * bytes: every weight row reads every activation row, and from PSRAM a whole
 * clip's activations (up to ~200 KB for the 2048-wide mHC input) do not stay
 * in the 64 KB data cache the two cores share. */
#define GEMM_TILE_BYTES (32 * 1024)

void nd_cq_gemm_rows(const nd_cact *c, const nd_tensor *t, const void *blob,
                     const float *xh, uint32_t T, uint32_t r0, uint32_t nrows,
                     float *y)
{
    gemm_ctx g;
    uint32_t in_pad = nd_cq_in_pad(t), i, k0;
    uint32_t tile = GEMM_TILE_BYTES / (in_pad * sizeof(float));
    float   *buf, *tab, *xt;

    if (nd_cq_int8 && t->group == 128 && (t->bits == 2 || t->bits == 4) &&
        in_pad <= 2048) {
        gemm8(c, t, blob, xh, T, r0, nrows, y);
        return;
    }
    if (tile < 1)
        tile = 1;
    if (tile > T)
        tile = T;
    buf = (float *)ND_QUANT_ALLOC((2 * in_pad + 1024 + (size_t)tile * in_pad) * sizeof(float));
    tab = buf + 2 * in_pad;
    xt  = tab + 1024;
    if (t->bits == 2) {
        const float *cb = nd_cact_codebook(c, 2);
        for (i = 0; i < 256; i++) {               /* LSB-first: index k at bits 2k */
            tab[4 * i]     = cb[i & 3];
            tab[4 * i + 1] = cb[(i >> 2) & 3];
            tab[4 * i + 2] = cb[(i >> 4) & 3];
            tab[4 * i + 3] = cb[(i >> 6) & 3];
        }
    }
    g.c       = c;
    g.t       = t;
    g.packed  = (const uint8_t *)blob;
    g.norms   = (const uint16_t *)(const void *)(g.packed +
                 (size_t)t->shape[0] * nd_cq_row_bytes(t));
    g.r0      = r0;
    g.ystride = nrows;
    g.wbuf[0] = buf;
    g.wbuf[1] = buf + in_pad;
    g.tab2    = tab;
    for (k0 = 0; k0 < T; k0 += tile) {
        uint32_t n = T - k0 < tile ? T - k0 : tile;
        memcpy(xt, xh + (size_t)k0 * in_pad, (size_t)n * in_pad * sizeof(float));
        g.xh = xt;
        g.y  = y + (size_t)k0 * nrows;
        g.T  = n;
        nd_parallel_rows(gemm_rows, &g, nrows);
    }
    ND_QUANT_FREE(buf);
}

void nd_cq_dequant_row(const nd_cact *c, const nd_tensor *t, const void *blob,
                       uint32_t row, float *scratch, float *w)
{
    const uint8_t  *packed   = (const uint8_t *)blob;
    uint32_t        out      = t->shape[0];
    uint32_t        g        = t->group;
    uint32_t        ngroup   = nd_cq_groups(t);
    uint32_t        rowbytes = nd_cq_row_bytes(t);
    const uint16_t *norms    = (const uint16_t *)(const void *)(packed +
                                (size_t)out * rowbytes);
    const float    *cb       = nd_cact_codebook(c, t->bits);
    const uint8_t  *p        = packed + (size_t)row * rowbytes;
    const uint16_t *nrm      = norms + (size_t)row * ngroup;
    float           scale    = 1.0f / sqrtf((float)g);
    uint32_t        gi;

    for (gi = 0; gi < ngroup; gi++) {
        float   *blk    = scratch + (size_t)gi * g;
        float    norm   = nd_f16(nrm[gi]);
        uint32_t bit_off = gi * g * t->bits;
        uint32_t j;

        for (j = 0; j < g; j++) {
            uint32_t b   = bit_off + j * t->bits;
            uint32_t byt = b >> 3;
            uint32_t sh  = b & 7u;
            uint32_t v   = p[byt];
            if (sh + t->bits > 8)
                v |= (uint32_t)p[byt + 1] << 8;
            blk[j] = cb[(v >> sh) & ((1u << t->bits) - 1u)] * norm;
        }
        nd_fwht(blk, g);
        for (j = 0; j < g; j++)
            blk[j] *= scale;
    }
    memcpy(w, scratch, t->shape[1] * sizeof(float));
}
