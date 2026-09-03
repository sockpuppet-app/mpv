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
// Every ring carries a generation. It is named in ring_changed(), in
// present(), and in the release call, and a slot is taken back with a
// compare-exchange against it, so a release belonging to a ring that has been
// superseded cannot free a slot of the live one.
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
    // A snapshot of the host's table and of the output description it named,
    // taken once in sp_init(). The VO thread reads these and never the
    // registration another thread wrote. host points at host_copy.
    mpv_sockpuppet_d3d11_host host_copy;
    mpv_sockpuppet_d3d11_host *host;
    DXGI_OUTPUT_DESC1 override_desc;
    bool have_override_desc;

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
    // The generation of the live ring. 0 before the first one exists.
    uint32_t generation;
    // Set when a ring could not be built for this description: the attempt is
    // not repeated, and not logged again, until the description changes.
    bool ring_failed;
    // Set when a keyed mutex was abandoned: the ring is rebuilt on the next
    // frame, because the surface and the mutex are no longer consistent.
    bool ring_dead;
    bool warned_stage_too_small;
    bool warned_display_changed;
    int64_t next_display_check_ns;

    // What the last colour space hint from vo_gpu_next asked for. libplacebo
    // answers a 10-bit swapchain with PQ for an HDR hint and with gamma 2.2
    // for an SDR one, and only the hint tells the two apart.
    bool hint_hdr;
    bool hint_wide;

    // Float16 scRGB for an HDR display, 8-bit sRGB for an SDR one, decided
    // once from the output before the swapchain exists. See sp_init().
    bool float16;

    // The time the frame in flight was copied out, on mpv's monotonic clock.
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
// A poll, not a wait. By contract the slot is free: its generation flag is 0,
// so the host is not holding the mutex. Waiting here would only turn a broken
// contract into a stall of the VO thread, which stalls presentation and A/V
// sync with it. A failure is information; take it and drop the frame.
#define KEYED_MUTEX_WAIT_MS 0

// A stage smaller than this is a host that has not sized itself yet. Nothing
// is copied out until it is real: a 1x1 shared texture is of no use to
// anybody, and pushing one at a compositor has gone badly before.
#define MIN_STAGE_SIZE 16

// How often the display is re-examined for a change between SDR and HDR.
#define DISPLAY_CHECK_INTERVAL_NS INT64_C(1000000000)

static void sp_uninit(struct ra_ctx *ctx);

static bool output_desc(struct priv *p, DXGI_OUTPUT_DESC1 *desc)
{
    if (p->have_override_desc) {
        *desc = p->override_desc;
        return true;
    }
    if (!p->host->window)
        return false;
    return mp_dxgi_output_desc_from_hwnd(&p->dxgi_ctx, (HWND)p->host->window,
                                         desc);
}

static bool display_is_hdr(struct priv *p)
{
    DXGI_OUTPUT_DESC1 desc;
    return output_desc(p, &desc) &&
           desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
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

// The swapchain format and libplacebo's bit depth were decided from the
// display once, in sp_init(), and there is no way to decide them again
// without building the gpu-next context again. Everything else about the
// target is re-read every frame, so a move between an SDR and an HDR display
// leaves the two disagreeing: tone mapping against a target the swapchain
// cannot carry, or scRGB into a display that wanted sRGB. Neither corrects
// itself. Say so once; the host's answer is to stop and start playback, which
// is what it already does when the window changes display.
static void check_display_change(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;

    if (p->warned_display_changed)
        return;
    int64_t now = mp_time_ns();
    if (now < p->next_display_check_ns)
        return;
    p->next_display_check_ns = now + DISPLAY_CHECK_INTERVAL_NS;

    if (display_is_hdr(p) == p->float16)
        return;
    p->warned_display_changed = true;
    MP_WARN(ctx, "The display changed between SDR and HDR while playing. The "
                 "swapchain format was decided when playback started and "
                 "cannot follow, so the picture is wrong until playback is "
                 "restarted.\n");
}

static int sp_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
    struct priv *p = ctx->priv;

    switch (request) {
    case VOCTRL_CHECK_EVENTS: {
        // The stage size is read here, at the top of every VO iteration.
        // mpv_sockpuppet_d3d11_wakeup() is what makes that iteration happen
        // now rather than at the next video frame, which on 24 fps content
        // is up to 42 ms away. The poll stays: it costs one host callback per
        // iteration, it is where the size has to be read either way, and it
        // is what makes a host that forgets to call, or one built against a
        // libmpv without the wakeup, converge anyway instead of freezing at
        // the old size.
        int w, h;
        read_size(ctx, &w, &h);
        if (w != ctx->vo->dwidth || h != ctx->vo->dheight) {
            ctx->vo->dwidth = w;
            ctx->vo->dheight = h;
            if (!resize(ctx))
                return VO_ERROR;
            *events |= VO_EVENT_RESIZE;
        }
        check_display_change(ctx);
        return VO_TRUE;
    }
    case VOCTRL_GET_DISPLAY_FPS: {
        // This is the refresh rate of the monitor the host's window is on.
        // The composition swapchain is attached to no visual, so whatever
        // DXGI paces Present against is the adapter's own output, which on a
        // mixed-refresh setup is not necessarily this one.
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

// close_handles is for a ring the host was never told about. Once
// ring_changed() has named the handles they are the host's to close, and
// closing one here would pull a texture out from under a frame in flight.
static void ring_free(struct priv *p, bool close_handles)
{
    for (int i = 0; i < MPV_SOCKPUPPET_D3D11_MAX_RING; i++) {
        SAFE_RELEASE(p->ring_mutex[i]);
        SAFE_RELEASE(p->ring[i]);
        if (close_handles && p->ring_handle[i])
            CloseHandle((HANDLE)p->ring_handle[i]);
        p->ring_handle[i] = NULL;
    }
    // The slot flags are deliberately left alone: they carry the generation
    // of the frame the host holds, and a release that names a generation that
    // is gone is refused by the compare-exchange in
    // mpv_sockpuppet_d3d11_release(). Clearing them here is what used to let
    // a stale release free a live slot of the ring that came next.
    p->ring_len = 0;
    p->ring_w = p->ring_h = 0;
    p->ring_fmt = DXGI_FORMAT_UNKNOWN;
    p->next_slot = 0;
}

// Tell the host that the ring it holds is gone: a count of zero supersedes
// every outstanding frame, and the generation goes with it so nothing that is
// still in flight can come back and land on a live slot later.
static void ring_notify_gone(struct priv *p)
{
    if (!p->ring_len)
        return;
    p->generation = 0;
    p->host->ring_changed(p->host->ctx, 0, 0, NULL, 0, 0, 0);
}

static uint32_t next_generation(struct priv *p)
{
    uint32_t gen = atomic_fetch_add(&p->state->generation, 1) + 1;
    // 0 is "no ring"; never hand it out as a live generation.
    if (!gen)
        gen = atomic_fetch_add(&p->state->generation, 1) + 1;
    return gen;
}

static bool ring_ensure(struct ra_ctx *ctx, const D3D11_TEXTURE2D_DESC *bd)
{
    struct priv *p = ctx->priv;

    bool same = p->ring_w == (int)bd->Width && p->ring_h == (int)bd->Height &&
                p->ring_fmt == bd->Format;
    if (p->ring_len && same && !p->ring_dead)
        return true;
    // Creating this ring already failed once. Retrying every frame only
    // repeats the error at frame rate; wait for the description to change.
    if (p->ring_failed && same)
        return false;

    ring_free(p, false);
    p->ring_dead = false;
    p->ring_failed = false;

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
    p->generation = next_generation(p);
    // These textures are new and nobody holds one. Any release still to come
    // for the ring before this one names an older generation and is refused.
    for (int i = 0; i < MPV_SOCKPUPPET_D3D11_MAX_RING; i++)
        atomic_store(&p->state->slot_gen[i], 0);
    MP_VERBOSE(ctx, "Shared texture ring %u: %d x %dx%d, DXGI format %d\n",
               (unsigned)p->generation, p->ring_len, p->ring_w, p->ring_h,
               (int)p->ring_fmt);
    p->host->ring_changed(p->host->ctx, p->generation, p->ring_len,
                          p->ring_handle, p->ring_w, p->ring_h,
                          (int)p->ring_fmt);
    return true;

error:
    // The host was never told about these handles, so they are nobody's:
    // close them, or an open NT handle keeps a full-size texture alive for
    // the life of the process.
    ring_free(p, true);
    p->ring_failed = true;
    p->ring_w = bd->Width;
    p->ring_h = bd->Height;
    p->ring_fmt = bd->Format;
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
        if (!atomic_load(&p->state->slot_gen[i])) {
            p->next_slot = (i + 1) % p->ring_len;
            return i;
        }
    }
    return -1;
}

static void frame_dropped(struct ra_ctx *ctx)
{
    // vo_gpu_next has no idea a frame vanished here, so without this the
    // stats page and vo-delayed-frame-count stay clean through a stall.
    vo_increment_drop_count(ctx->vo, 1);
}

// The finished frame is in back buffer 0 between libplacebo's submit and the
// Present in sp_swap_buffers, which is exactly where vo_gpu_next calls this.
// Returns the slot the frame was copied into, or -1, and fills *out_csp for
// the caller to hand to the host after the Present.
static int copy_frame_out(struct ra_ctx *ctx, int *out_csp)
{
    struct priv *p = ctx->priv;
    ID3D11Texture2D *back = NULL;
    int slot = -1;
    HRESULT hr;

    hr = IDXGISwapChain_GetBuffer(p->swapchain, 0, &IID_ID3D11Texture2D,
                                  (void **)&back);
    if (FAILED(hr)) {
        MP_ERR(ctx, "Couldn't get the back buffer: %s\n", mp_HRESULT_to_str(hr));
        frame_dropped(ctx);
        return -1;
    }

    D3D11_TEXTURE2D_DESC bd;
    ID3D11Texture2D_GetDesc(back, &bd);

    if (bd.Width < MIN_STAGE_SIZE || bd.Height < MIN_STAGE_SIZE) {
        if (!p->warned_stage_too_small) {
            p->warned_stage_too_small = true;
            MP_WARN(ctx, "The host's stage is %ux%u; nothing is copied out "
                         "until it is at least %dx%d.\n", bd.Width, bd.Height,
                    MIN_STAGE_SIZE, MIN_STAGE_SIZE);
        }
        goto done;
    }
    p->warned_stage_too_small = false;

    if (!ring_ensure(ctx, &bd)) {
        frame_dropped(ctx);
        goto done;
    }

    slot = pick_slot(p);
    if (slot < 0) {
        MP_VERBOSE(ctx, "The host holds every shared texture; frame dropped.\n");
        frame_dropped(ctx);
        goto done;
    }

    hr = IDXGIKeyedMutex_AcquireSync(p->ring_mutex[slot], KEYED_MUTEX_KEY,
                                     KEYED_MUTEX_WAIT_MS);
    if (hr != S_OK) {
        if (hr == (HRESULT)WAIT_ABANDONED) {
            // The surface and its mutex are no longer in a consistent state
            // and the only documented answer is to recreate both. Dropping
            // the frame and coming back to the same slot leaves it dead for
            // the life of the ring. Ownership is granted with an abandoned
            // mutex, so give it back before the ring goes.
            MP_WARN(ctx, "The keyed mutex of shared texture %d was abandoned; "
                         "rebuilding the ring.\n", slot);
            IDXGIKeyedMutex_ReleaseSync(p->ring_mutex[slot], KEYED_MUTEX_KEY);
            p->ring_dead = true;
        } else if (hr != (HRESULT)WAIT_TIMEOUT) {
            // Not a timeout and not an abandonment: the mutex is not usable.
            // Nothing is owned here, so nothing is released.
            MP_WARN(ctx, "Couldn't acquire the keyed mutex of shared texture "
                         "%d: %s. Rebuilding the ring.\n", slot,
                    mp_HRESULT_to_str(hr));
            p->ring_dead = true;
        } else {
            MP_VERBOSE(ctx, "Shared texture %d is still being read; frame "
                            "dropped.\n", slot);
        }
        slot = -1;
        frame_dropped(ctx);
        goto done;
    }
    ID3D11DeviceContext_CopyResource(p->imm, (ID3D11Resource *)p->ring[slot],
                                     (ID3D11Resource *)back);
    // The release has to be recorded before the flush, or it is left sitting
    // on the immediate context and only the next Present submits it: the host
    // would be told the frame is ready while the GPU still has mpv holding
    // the mutex. One submit does for both.
    IDXGIKeyedMutex_ReleaseSync(p->ring_mutex[slot], KEYED_MUTEX_KEY);
    ID3D11DeviceContext_Flush(p->imm);

    *out_csp = color_space_kind(p, bd.Format);
    // The slot is the host's from here. The callback comes after the Present.
    atomic_store(&p->state->slot_gen[slot], p->generation);

done:
    SAFE_RELEASE(back);
    return slot;
}

static void sp_swap_buffers(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;
    int csp = 0;

    // vo_gpu_next never calls the swapchain's submit_frame hook, so there is
    // no struct vo_frame to be had here and no media pts with it. What the
    // host is given is the time of the copy on mpv's monotonic clock, which
    // is what it needs: a strictly increasing frame timeline.
    p->frame_pts = mp_time_ns();

    int slot = copy_frame_out(sw->ctx, &csp);

    LARGE_INTEGER perf_count;
    QueryPerformanceCounter(&perf_count);
    p->last_submit_qpc = perf_count.QuadPart;

    // Nothing shows this, but DXGI still paces it to the display, which is
    // what vo_gpu_next expects of swap_buffers: a frame carrying video waits
    // for a vsync, and that wait is the whole of this context's frame timing.
    //
    // Except for a redraw the host asked for, which is presented with a sync
    // interval of 0 and does not wait. The picture is already out: the copy
    // into the shared texture above is what the host will see, and the
    // Present that follows it shows nothing to anybody. All the wait does to
    // a redraw is hold the VO thread until DXGI's queue drains, and that
    // queue is the adapter's own output, roughly sixty slots a second
    // however fast the panel runs. A host animating a property at the
    // display's rate asks for redraws faster than that, and while a film is
    // playing its video frames have first claim on the slots, so most of the
    // redraws never became a picture at all. See redraw_pending.
    bool asked_for = atomic_exchange(&p->state->redraw_pending, false);
    IDXGISwapChain_Present(p->swapchain, asked_for ? 0 : 1, 0);

    // After the Present: by now the mutex release the copy recorded has
    // certainly been submitted, so a host that acquires the moment it hears
    // about the frame cannot fail or stall on it.
    if (slot >= 0)
        p->host->present(p->host->ctx, p->generation, slot, p->frame_pts, csp);
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

    // Retract the VO first, and under the lock, so that a wakeup already
    // inside vo_redraw() finishes before anything here runs and no wakeup
    // that arrives afterwards finds a pointer at all. The comparison matters
    // for the failure path: sp_init() calls this before it has published,
    // and must not clear a VO it never set.
    if (p->state) {
        mp_mutex_lock(&p->state->vo_lock);
        if (p->state->vo == ctx->vo) {
            p->state->vo = NULL;
            // A wakeup can set the mark and then find no frame to be drawn
            // before the VO goes. Clearing it here, under the lock that
            // guards the pointer, is what keeps a mark from outliving the VO
            // it was meant for and making one frame of the next one unsynced.
            atomic_store(&p->state->redraw_pending, false);
        }
        mp_mutex_unlock(&p->state->vo_lock);
    }

    // Tell the host before anything is torn down: it is holding NT handles to
    // a device that is about to be destroyed, and nothing else would ever
    // say so.
    if (p->host)
        ring_notify_gone(p);
    ring_free(p, false);
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
    // No submit_frame: vo_gpu_next never calls the hook, and start_frame
    // refuses vo_gpu, which is the only caller of it in the tree.
    .swap_buffers    = sp_swap_buffers,
    .get_vsync       = sp_get_vsync,
};

// The LUID of the adapter a device was actually created on.
static bool device_adapter_luid(ID3D11Device *dev, uint64_t *out)
{
    IDXGIDevice1 *dxgi_dev = NULL;
    IDXGIAdapter1 *adapter = NULL;
    DXGI_ADAPTER_DESC1 desc;
    bool ok = false;

    if (FAILED(ID3D11Device_QueryInterface(dev, &IID_IDXGIDevice1,
                                           (void **)&dxgi_dev)))
        goto done;
    if (FAILED(IDXGIDevice1_GetParent(dxgi_dev, &IID_IDXGIAdapter1,
                                      (void **)&adapter)))
        goto done;
    if (FAILED(IDXGIAdapter1_GetDesc1(adapter, &desc)))
        goto done;
    *out = (uint64_t)(uint32_t)desc.AdapterLuid.HighPart << 32 |
           (uint32_t)desc.AdapterLuid.LowPart;
    ok = true;

done:
    SAFE_RELEASE(adapter);
    SAFE_RELEASE(dxgi_dev);
    return ok;
}

static bool sp_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv = talloc_zero(ctx, struct priv);

    p->state = ctx->global->sockpuppet_d3d11;
    if (!p->state || !atomic_load(&p->state->registered)) {
        MP_MSG(ctx, ctx->vo->probing ? MSGL_V : MSGL_ERR,
               "No host: call mpv_sockpuppet_d3d11_set_host() before the VO "
               "is created.\n");
        p->state = NULL;
        return false;
    }
    // A snapshot. The registration is written once, before it is published,
    // and a second registration that differs is refused, so nothing the VO
    // thread reads from here can change under it.
    p->host_copy = p->state->host;
    p->host = &p->host_copy;
    p->override_desc = p->state->override_desc;
    p->have_override_desc = p->state->have_override_desc;

    // What --d3d11-output-format=auto decides for a window: float16 scRGB
    // when the display is HDR, 8 bits of sRGB when it is not. Float16 into
    // an SDR display was measured and is not the same picture: libplacebo
    // maps SDR content into a linear target that carries the monitor's
    // luminance metadata, and white came out 2.4% darker with the blacks
    // lifted, before the host had touched the texture. Into an sRGB target
    // the mapping is the identity the windowed path has always been, and
    // the host reads 8-bit RGBA as readily as float16. The HDR display keeps
    // float16, which is the scRGB mpv presents with rgba16f.
    p->float16 = display_is_hdr(p);

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

    // mp_d3d11_create_present_device() falls back to the default adapter when
    // the LUID matches nothing, which is right for a window and wrong here: a
    // keyed-mutex NT-handle texture cannot be opened by a device on another
    // adapter, so the host would fail to import every frame and show nothing.
    // Refuse instead, and say which LUID was asked for, because the usual
    // cause is a host that sign-extended the high half.
    if (p->host->adapter_luid) {
        uint64_t got = 0;
        if (!device_adapter_luid(p->device, &got) ||
            got != p->host->adapter_luid)
        {
            MP_FATAL(ctx, "The host asked for the adapter with LUID %016llx "
                          "and the device was created on %016llx. A shared "
                          "texture cannot cross adapters, so there is nothing "
                          "to render into.\n",
                     (unsigned long long)p->host->adapter_luid,
                     (unsigned long long)got);
            goto error;
        }
    }
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
        .skip_output_probe = true,
        .configured_csp = &p->swapchain_csp,
        .flip = true,
        // Add one frame for the backbuffer
        .length = ctx->vo->opts->swapchain_depth + 1,
        .usage = usage,
    };
    if (!mp_d3d11_create_swapchain(p->device, ctx->log, &scopts, &p->swapchain))
        goto error;

    // Last, once nothing else can fail: from here the host may wake this VO
    // through mpv_sockpuppet_d3d11_wakeup(). Published under the lock the
    // wakeup takes, so it never sees a half-built context, and retracted the
    // same way in sp_uninit().
    mp_mutex_lock(&p->state->vo_lock);
    p->state->vo = ctx->vo;
    mp_mutex_unlock(&p->state->vo_lock);

    return true;

error:
    sp_uninit(ctx);
    return false;
}

IDXGISwapChain *ra_d3d11_sockpuppet_get_swapchain(struct ra_ctx *ra)
{
    struct priv *p = ra->priv;
    if (!p || !p->swapchain)
        return NULL;
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
    // 10-bit swapchain for SDR, and no host of this context opens a 10-bit
    // shared handle on any display. Unconditional, not keyed on the display,
    // because what cannot be imported is the format and not the monitor.
    params->disable_10bit_sdr = true;
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
