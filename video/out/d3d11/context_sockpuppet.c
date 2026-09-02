/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

// The sockpuppet-d3d11 render context: Direct3D 11 with no window.
//
// A host that embeds libmpv registers itself with
// mpv_sockpuppet_d3d11_set_host() and names the size of a stage. This context
// creates the device the way the windowed d3d11 context does, on the adapter
// the host names, and renders into a composition swapchain on it, exactly as
// --d3d11-output-mode=composition does. Nothing is attached to that
// swapchain: Present still runs so DXGI paces the render loop to the display,
// but no window ever shows its buffers. Instead, after every frame, the
// finished back buffer is copied on the GPU into one of a ring of shared
// textures and the host is told which one. The host imports that texture
// wherever it likes; when it is done it hands the slot back with
// mpv_sockpuppet_d3d11_release().
//
// Everything that decides the picture stays where it is: vo_gpu_next.c and
// libplacebo render and negotiate the swapchain format and colour space
// unchanged, and the target colour space, depth and SDR white level are read
// from the monitor the host's window is on through the same helpers the
// windowed context uses. The copy is byte-exact.

#include <stdatomic.h>
#include <dxgi1_2.h>

#include "common/global.h"
#include "common/msg.h"
#include "osdep/timer.h"
#include "osdep/windows_utils.h"

#include "video/out/gpu/context.h"
#include "video/out/gpu/d3d11_helpers.h"
#include "video/out/gpu/spirv.h"
#include "context.h"
#include "context_sockpuppet.h"
#include "ra_d3d11.h"

#ifdef PL_HAVE_D3D11
#include <libplacebo/d3d11.h>
#endif

struct priv {
    struct mp_sockpuppet_d3d11 *state;
    mpv_sockpuppet_d3d11_host *host;

    ID3D11Device *device;
    ID3D11DeviceContext *imm;
    IDXGISwapChain *swapchain;
    struct pl_color_space swapchain_csp;
    struct mp_dxgi_factory_ctx dxgi_ctx;

    // The ring of shared textures the finished frames are copied into. The
    // handles belong to the host once ring_changed() has named them.
    int ring_len;
    ID3D11Texture2D *ring[MPV_SOCKPUPPET_D3D11_MAX_RING];
    IDXGIKeyedMutex *ring_mutex[MPV_SOCKPUPPET_D3D11_MAX_RING];
    void *ring_handle[MPV_SOCKPUPPET_D3D11_MAX_RING];
    int ring_w, ring_h;
    DXGI_FORMAT ring_fmt;
    int next_slot;

    // What the last colour space hint from vo_gpu_next asked for. libplacebo
    // answers a 10-bit swapchain with PQ for an HDR hint and with gamma 2.2
    // for an SDR one, and only the hint tells the two apart.
    bool hint_hdr;
    bool hint_wide;

    // Float16 scRGB for an HDR display, 8-bit sRGB for an SDR one, decided
    // once from the output before the swapchain exists. See sp_init().
    bool float16;

    // The pts of the frame between submit_frame and swap_buffers.
    int64_t frame_pts;

    int64_t perf_freq;
    unsigned last_sync_refresh_count;
    int64_t last_sync_qpc_time;
    int64_t vsync_duration_qpc;
    int64_t last_submit_qpc;
};

// The keyed mutex key both sides use. Chromium's shared-image backing
// acquires and releases key 0, so the mutex works as a plain mutex.
#define KEYED_MUTEX_KEY 0
#define KEYED_MUTEX_WAIT_MS 200

static void sp_uninit(struct ra_ctx *ctx);

static bool output_desc(struct priv *p, DXGI_OUTPUT_DESC1 *desc)
{
    if (p->host->override_output_desc) {
        *desc = *(const DXGI_OUTPUT_DESC1 *)p->host->override_output_desc;
        return true;
    }
    if (!p->host->window)
        return false;
    return mp_dxgi_output_desc_from_hwnd(&p->dxgi_ctx, (HWND)p->host->window,
                                         desc);
}

static void read_size(struct ra_ctx *ctx, int *w, int *h)
{
    struct priv *p = ctx->priv;
    *w = *h = 0;
    p->host->get_size(p->host->ctx, w, h);
    *w = MPMAX(*w, 1);
    *h = MPMAX(*h, 1);
}

static bool resize(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    HRESULT hr = IDXGISwapChain_ResizeBuffers(p->swapchain, 0, ctx->vo->dwidth,
        ctx->vo->dheight, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) {
        MP_FATAL(ctx, "Couldn't resize swapchain: %s\n", mp_HRESULT_to_str(hr));
        return false;
    }
    return true;
}

static bool sp_reconfig(struct ra_ctx *ctx)
{
    read_size(ctx, &ctx->vo->dwidth, &ctx->vo->dheight);
    return resize(ctx);
}

static int sp_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
    struct priv *p = ctx->priv;

    switch (request) {
    case VOCTRL_CHECK_EVENTS: {
        // The host has no way to wake the VO thread, so the stage size is
        // polled here, which the VO loop reaches often.
        int w, h;
        read_size(ctx, &w, &h);
        if (w != ctx->vo->dwidth || h != ctx->vo->dheight) {
            ctx->vo->dwidth = w;
            ctx->vo->dheight = h;
            if (!resize(ctx))
                return VO_ERROR;
            *events |= VO_EVENT_RESIZE;
        }
        return VO_TRUE;
    }
    case VOCTRL_GET_DISPLAY_FPS: {
        DXGI_OUTPUT_DESC1 desc;
        if (!output_desc(p, &desc))
            return VO_FALSE;
        DEVMODEW mode = { .dmSize = sizeof(mode) };
        if (!EnumDisplaySettingsW(desc.DeviceName, ENUM_CURRENT_SETTINGS, &mode) ||
            !mode.dmDisplayFrequency)
            return VO_FALSE;
        *(double *)arg = mode.dmDisplayFrequency;
        return VO_TRUE;
    }
    case VOCTRL_GET_DISPLAY_RES: {
        DXGI_OUTPUT_DESC1 desc;
        if (!output_desc(p, &desc))
            return VO_FALSE;
        ((int *)arg)[0] = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
        ((int *)arg)[1] = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
        return VO_TRUE;
    }
    case VOCTRL_GET_HIDPI_SCALE: {
        float scale = p->host->get_dpi_scale ? p->host->get_dpi_scale(p->host->ctx) : 0;
        *(double *)arg = scale > 0 ? scale : 1.0;
        return VO_TRUE;
    }
    }
    return VO_NOTIMPL;
}

static int sp_color_depth(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;

    DXGI_OUTPUT_DESC1 desc1;
    if (!output_desc(p, &desc1))
        desc1.BitsPerColor = 0;

    DXGI_SWAP_CHAIN_DESC desc;
    HRESULT hr = IDXGISwapChain_GetDesc(p->swapchain, &desc);
    if (FAILED(hr)) {
        MP_ERR(sw->ctx, "Failed to query swap chain description: %s!\n",
               mp_HRESULT_to_str(hr));
        return desc1.BitsPerColor;
    }

    const struct ra_format *ra_fmt =
        ra_d3d11_get_ra_format(sw->ctx->ra, desc.BufferDesc.Format);
    if (!ra_fmt || !ra_fmt->component_depth[0])
        return desc1.BitsPerColor;

    if (!desc1.BitsPerColor)
        return ra_fmt->component_depth[0];

    return MPMIN(ra_fmt->component_depth[0], desc1.BitsPerColor);
}

static struct pl_color_space sp_target_color_space(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;

    DXGI_OUTPUT_DESC1 desc;
    if (output_desc(p, &desc))
        return mp_dxgi_desc_to_color_space(&desc);

    return (struct pl_color_space){0};
}

static float sp_target_ref_luma(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;

    if (p->host->override_sdr_white_level > 0)
        return p->host->override_sdr_white_level;
    if (!p->host->window)
        return 0;
    return mp_dxgi_sdr_white_level_from_hwnd(&p->dxgi_ctx, (HWND)p->host->window);
}

static bool sp_set_color(struct ra_swapchain *sw, struct mp_image_params *params)
{
    struct priv *p = sw->priv;

    // Not handled here: libplacebo configures the swapchain from the same
    // hint. Only what it asked for is remembered, to name the colour space
    // of a 10-bit copy to the host.
    p->hint_hdr = params && pl_color_transfer_is_hdr(params->color.transfer);
    p->hint_wide = params && pl_color_primaries_is_wide_gamut(params->color.primaries);
    return false;
}

static bool sp_start_frame(struct ra_swapchain *sw, struct ra_fbo *out_fbo)
{
    if (!out_fbo)
        return true;

    MP_ERR(sw->ctx, "The sockpuppet-d3d11 context renders through libplacebo "
                    "only; use --vo=gpu-next.\n");
    return false;
}

static bool sp_submit_frame(struct ra_swapchain *sw, const struct vo_frame *frame)
{
    struct priv *p = sw->priv;
    p->frame_pts = frame ? frame->pts : 0;
    return true;
}

static void ring_free(struct priv *p)
{
    for (int i = 0; i < p->ring_len; i++) {
        SAFE_RELEASE(p->ring_mutex[i]);
        SAFE_RELEASE(p->ring[i]);
        // The handle is the host's to close once ring_changed() named it.
        p->ring_handle[i] = NULL;
        atomic_store(&p->state->busy[i], 0);
    }
    p->ring_len = 0;
    p->ring_w = p->ring_h = 0;
    p->ring_fmt = DXGI_FORMAT_UNKNOWN;
    p->next_slot = 0;
}

static bool ring_ensure(struct ra_ctx *ctx, const D3D11_TEXTURE2D_DESC *bd)
{
    struct priv *p = ctx->priv;

    if (p->ring_len && p->ring_w == (int)bd->Width && p->ring_h == (int)bd->Height &&
        p->ring_fmt == bd->Format)
        return true;

    ring_free(p);

    int n = MPCLAMP(p->host->ring_length, 2, MPV_SOCKPUPPET_D3D11_MAX_RING);
    for (int i = 0; i < n; i++) {
        D3D11_TEXTURE2D_DESC desc = {
            .Width = bd->Width,
            .Height = bd->Height,
            .MipLevels = 1,
            .ArraySize = 1,
            .Format = bd->Format,
            .SampleDesc = { .Count = 1 },
            .Usage = D3D11_USAGE_DEFAULT,
            .BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
            .MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                         D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
        };
        HRESULT hr = ID3D11Device_CreateTexture2D(p->device, &desc, NULL, &p->ring[i]);
        if (FAILED(hr)) {
            MP_ERR(ctx, "Couldn't create shared texture %d: %s\n", i,
                   mp_HRESULT_to_str(hr));
            goto error;
        }
        p->ring_len = i + 1;

        hr = ID3D11Texture2D_QueryInterface(p->ring[i], &IID_IDXGIKeyedMutex,
                                            (void **)&p->ring_mutex[i]);
        if (FAILED(hr)) {
            MP_ERR(ctx, "Shared texture %d has no keyed mutex: %s\n", i,
                   mp_HRESULT_to_str(hr));
            goto error;
        }

        IDXGIResource1 *res = NULL;
        hr = ID3D11Texture2D_QueryInterface(p->ring[i], &IID_IDXGIResource1,
                                            (void **)&res);
        if (FAILED(hr)) {
            MP_ERR(ctx, "Shared texture %d is not a DXGI resource: %s\n", i,
                   mp_HRESULT_to_str(hr));
            goto error;
        }
        HANDLE handle = NULL;
        hr = IDXGIResource1_CreateSharedHandle(res, NULL,
            DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &handle);
        SAFE_RELEASE(res);
        if (FAILED(hr)) {
            MP_ERR(ctx, "Couldn't share texture %d: %s\n", i,
                   mp_HRESULT_to_str(hr));
            goto error;
        }
        p->ring_handle[i] = handle;
    }

    p->ring_w = bd->Width;
    p->ring_h = bd->Height;
    p->ring_fmt = bd->Format;
    p->next_slot = 0;
    MP_VERBOSE(ctx, "Shared texture ring: %d x %dx%d, DXGI format %d\n",
               p->ring_len, p->ring_w, p->ring_h, (int)p->ring_fmt);
    p->host->ring_changed(p->host->ctx, p->ring_len, p->ring_handle,
                          p->ring_w, p->ring_h, (int)p->ring_fmt);
    return true;

error:
    ring_free(p);
    return false;
}

// Float16 is scRGB and 8-bit is sRGB; the 10-bit cases are kept for a host
// that asks for something else one day.
static int color_space_kind(struct priv *p, DXGI_FORMAT fmt)
{
    switch (fmt) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return MPV_SOCKPUPPET_D3D11_CSP_SCRGB_LINEAR;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        if (p->hint_hdr)
            return MPV_SOCKPUPPET_D3D11_CSP_PQ;
        if (p->hint_wide)
            return MPV_SOCKPUPPET_D3D11_CSP_BT2020_G22;
        return MPV_SOCKPUPPET_D3D11_CSP_SRGB;
    default:
        return MPV_SOCKPUPPET_D3D11_CSP_SRGB;
    }
}

static int pick_slot(struct priv *p)
{
    for (int k = 0; k < p->ring_len; k++) {
        int i = (p->next_slot + k) % p->ring_len;
        if (!atomic_load(&p->state->busy[i])) {
            p->next_slot = (i + 1) % p->ring_len;
            return i;
        }
    }
    return -1;
}

// The finished frame is in back buffer 0 between libplacebo's submit and the
// Present below, which is exactly where vo_gpu_next calls this.
static void copy_frame_out(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    ID3D11Texture2D *back = NULL;
    HRESULT hr;

    hr = IDXGISwapChain_GetBuffer(p->swapchain, 0, &IID_ID3D11Texture2D,
                                  (void **)&back);
    if (FAILED(hr)) {
        MP_ERR(ctx, "Couldn't get the back buffer: %s\n", mp_HRESULT_to_str(hr));
        return;
    }

    D3D11_TEXTURE2D_DESC bd;
    ID3D11Texture2D_GetDesc(back, &bd);
    if (!ring_ensure(ctx, &bd))
        goto done;

    int slot = pick_slot(p);
    if (slot < 0) {
        MP_VERBOSE(ctx, "The host holds every shared texture; frame dropped.\n");
        goto done;
    }

    hr = IDXGIKeyedMutex_AcquireSync(p->ring_mutex[slot], KEYED_MUTEX_KEY,
                                     KEYED_MUTEX_WAIT_MS);
    if (hr != S_OK) {
        MP_VERBOSE(ctx, "Shared texture %d is still being read; frame dropped.\n",
                   slot);
        goto done;
    }
    ID3D11DeviceContext_CopyResource(p->imm, (ID3D11Resource *)p->ring[slot],
                                     (ID3D11Resource *)back);
    ID3D11DeviceContext_Flush(p->imm);
    IDXGIKeyedMutex_ReleaseSync(p->ring_mutex[slot], KEYED_MUTEX_KEY);

    atomic_store(&p->state->busy[slot], 1);
    p->host->present(p->host->ctx, slot, p->frame_pts,
                     color_space_kind(p, bd.Format));

done:
    SAFE_RELEASE(back);
}

static void sp_swap_buffers(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;

    copy_frame_out(sw->ctx);

    LARGE_INTEGER perf_count;
    QueryPerformanceCounter(&perf_count);
    p->last_submit_qpc = perf_count.QuadPart;

    // Nothing shows this, but DXGI still paces it to the display, which is
    // what vo_gpu_next expects of swap_buffers.
    IDXGISwapChain_Present(p->swapchain, 1, 0);
}

static int64_t qpc_to_ns(struct ra_swapchain *sw, int64_t qpc)
{
    struct priv *p = sw->priv;
    return qpc / p->perf_freq * INT64_C(1000000000) +
        qpc % p->perf_freq * INT64_C(1000000000) / p->perf_freq;
}

static int64_t qpc_ns_now(struct ra_swapchain *sw)
{
    LARGE_INTEGER perf_count;
    QueryPerformanceCounter(&perf_count);
    return qpc_to_ns(sw, perf_count.QuadPart);
}

// Verbatim from the windowed context, which presents on every vsync as this
// one does.
static void sp_get_vsync(struct ra_swapchain *sw, struct vo_vsync_info *info)
{
    struct priv *p = sw->priv;
    HRESULT hr;

    UINT submit_count;
    hr = IDXGISwapChain_GetLastPresentCount(p->swapchain, &submit_count);
    if (FAILED(hr))
        return;

    DXGI_FRAME_STATISTICS stats;
    hr = IDXGISwapChain_GetFrameStatistics(p->swapchain, &stats);
    if (hr == DXGI_ERROR_FRAME_STATISTICS_DISJOINT) {
        p->last_sync_refresh_count = 0;
        p->last_sync_qpc_time = 0;
    }
    if (FAILED(hr))
        return;

    info->skipped_vsyncs = -1;

    unsigned src_passed = 0;
    if (stats.SyncRefreshCount && p->last_sync_refresh_count)
        src_passed = stats.SyncRefreshCount - p->last_sync_refresh_count;
    p->last_sync_refresh_count = stats.SyncRefreshCount;

    unsigned sqt_passed = 0;
    if (stats.SyncQPCTime.QuadPart && p->last_sync_qpc_time)
        sqt_passed = stats.SyncQPCTime.QuadPart - p->last_sync_qpc_time;
    p->last_sync_qpc_time = stats.SyncQPCTime.QuadPart;

    if (src_passed && sqt_passed)
        p->vsync_duration_qpc = sqt_passed / src_passed;
    if (p->vsync_duration_qpc)
        info->vsync_duration = qpc_to_ns(sw, p->vsync_duration_qpc);

    if (p->vsync_duration_qpc && stats.PresentCount &&
        stats.PresentRefreshCount && stats.SyncRefreshCount &&
        stats.SyncQPCTime.QuadPart)
    {
        unsigned expected_sync_pc = stats.PresentCount +
            (stats.SyncRefreshCount - stats.PresentRefreshCount);
        int queued_frames = submit_count - expected_sync_pc;
        int64_t last_queue_display_time_qpc = stats.SyncQPCTime.QuadPart +
            queued_frames * p->vsync_duration_qpc;
        if (last_queue_display_time_qpc >= p->last_submit_qpc) {
            info->last_queue_display_time = mp_time_ns() +
                (qpc_to_ns(sw, last_queue_display_time_qpc) - qpc_ns_now(sw));
        }
    }
}

static void sp_uninit(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;

    ring_free(p);
    SAFE_RELEASE(p->swapchain);
    SAFE_RELEASE(p->imm);
    SAFE_RELEASE(p->device);
    mp_dxgi_factory_uninit(&p->dxgi_ctx);

    // Destroy the RA last to prevent objects we hold from showing up in D3D's
    // leak checker
    if (ctx->ra)
        ctx->ra->fns->destroy(ctx->ra);
}

static const struct ra_swapchain_fns sp_swapchain = {
    .color_depth     = sp_color_depth,
    .target_csp      = sp_target_color_space,
    .target_ref_luma = sp_target_ref_luma,
    .set_color       = sp_set_color,
    .start_frame     = sp_start_frame,
    .submit_frame    = sp_submit_frame,
    .swap_buffers    = sp_swap_buffers,
    .get_vsync       = sp_get_vsync,
};

static bool sp_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv = talloc_zero(ctx, struct priv);

    p->state = ctx->global->sockpuppet_d3d11;
    if (!p->state) {
        MP_MSG(ctx, ctx->vo->probing ? MSGL_V : MSGL_ERR,
               "No host: call mpv_sockpuppet_d3d11_set_host() before the VO "
               "is created.\n");
        return false;
    }
    p->host = &p->state->host;

    // What --d3d11-output-format=auto decides for a window: float16 scRGB
    // when the display is HDR, 8 bits of sRGB when it is not. Float16 into
    // an SDR display was measured and is not the same picture: libplacebo
    // maps SDR content into a linear target that carries the monitor's
    // luminance metadata, and white came out 2.4% darker with the blacks
    // lifted, before the host had touched the texture. Into an sRGB target
    // the mapping is the identity the windowed path has always been, and
    // the host reads 8-bit RGBA as readily as float16. The HDR display keeps
    // float16, which is the scRGB mpv presents with rgba16f.
    {
        DXGI_OUTPUT_DESC1 desc;
        p->float16 = output_desc(p, &desc) &&
                     desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    }

    LARGE_INTEGER perf_freq;
    QueryPerformanceFrequency(&perf_freq);
    p->perf_freq = perf_freq.QuadPart;

    struct ra_swapchain *sw = ctx->swapchain = talloc_zero(ctx, struct ra_swapchain);
    sw->priv = p;
    sw->ctx = ctx;
    sw->fns = &sp_swapchain;

    // The device is created the way the windowed context creates its own,
    // so hardware decoding shares it and libplacebo wraps it the same way.
    struct d3d11_device_opts dopts = {
        .debug = ctx->opts.debug,
        .allow_warp = false,
        .force_warp = false,
        .max_feature_level = D3D_FEATURE_LEVEL_12_1,
        .max_frame_latency = ctx->vo->opts->swapchain_depth,
        .adapter_luid = p->host->adapter_luid,
    };
    if (!mp_d3d11_create_present_device(ctx->log, &dopts, &p->device))
        goto error;
    ID3D11Device_GetImmediateContext(p->device, &p->imm);

    if (!spirv_compiler_init(ctx))
        goto error;
    ctx->ra = ra_d3d11_create(p->device, ctx->log, ctx->spirv);
    if (!ctx->ra)
        goto error;

    // There is no window. Nothing in the gpu-next path consults w32 when
    // this is set, and the swapchain below is a composition one.
    ctx->opts.composition = true;
    read_size(ctx, &ctx->vo->dwidth, &ctx->vo->dheight);

    UINT usage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_SHADER_INPUT;
    if (ID3D11Device_GetFeatureLevel(p->device) >= D3D_FEATURE_LEVEL_11_0)
        usage |= DXGI_USAGE_UNORDERED_ACCESS;

    struct d3d11_swapchain_opts scopts = {
        .window = NULL,
        .width = ctx->vo->dwidth,
        .height = ctx->vo->dheight,
        // Decided here, not probed: a composition swapchain has no output to
        // ask. See the decision in sp_init() and
        // ra_d3d11_sockpuppet_set_swapchain_params().
        .format = p->float16 ? DXGI_FORMAT_R16G16B16A16_FLOAT
                             : DXGI_FORMAT_R8G8B8A8_UNORM,
        .color_space = p->float16 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                                  : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        .configured_csp = &p->swapchain_csp,
        .flip = true,
        // Add one frame for the backbuffer
        .length = ctx->vo->opts->swapchain_depth + 1,
        .usage = usage,
    };
    if (!mp_d3d11_create_swapchain(p->device, ctx->log, &scopts, &p->swapchain))
        goto error;

    return true;

error:
    sp_uninit(ctx);
    return false;
}

IDXGISwapChain *ra_d3d11_sockpuppet_get_swapchain(struct ra_ctx *ra)
{
    struct priv *p = ra->priv;
    IDXGISwapChain_AddRef(p->swapchain);
    return p->swapchain;
}

#ifdef PL_HAVE_D3D11
void ra_d3d11_sockpuppet_set_swapchain_params(struct ra_ctx *ra,
                                              struct pl_d3d11_swapchain_params *params)
{
    // Left to its own choice libplacebo picks a 10-bit swapchain for SDR as
    // well as for HDR, and the host that reads these textures (Chromium)
    // opens a shared handle only as 8-bit RGBA or BGRA, or as float16. So
    // it is told: 16 bits for an HDR display, which libplacebo pairs with
    // scRGB, and 8 for an SDR one, which it pairs with sRGB and renders the
    // way the windowed context does. See sp_init() for the measurement.
    struct priv *p = ra->priv;
    params->color_bits = p->float16 ? 16 : 8;
    params->alpha_bits = 0;
    // color_bits is a floor, not a choice: left to itself libplacebo takes a
    // 10-bit swapchain for SDR, and Chromium opens no 10-bit shared handle.
    params->disable_10bit_sdr = !p->float16;
}
#endif

const struct ra_ctx_fns ra_ctx_d3d11_sockpuppet = {
    .type        = "d3d11",
    .name        = "sockpuppet-d3d11",
    .description = "Direct3D 11 into shared textures for an embedding host",
    .reconfig    = sp_reconfig,
    .control     = sp_control,
    .init        = sp_init,
    .uninit      = sp_uninit,
};
