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

#pragma once

#include <stdatomic.h>
#include <dxgi.h>
#include <dxgi1_6.h>

#include "mpv/client.h"
#include "osdep/threads.h"
#include "video/out/gpu/context.h"

// What mpv_sockpuppet_d3d11_set_host() registers on the mpv_global, and what
// the sockpuppet-d3d11 render context reads. One per core, allocated by
// mp_clients_init() on the core thread so nothing has to publish a pointer
// across threads later.
struct mp_sockpuppet_d3d11 {
    // Written once, by the first mpv_sockpuppet_d3d11_set_host() under
    // mp_client_api.lock; published by the store to registered below. A
    // second call with different fields is refused, so the VO thread reads
    // an immutable table.
    mpv_sockpuppet_d3d11_host host;
    // A copy of what host.override_output_desc pointed at when the host
    // registered, so the VO thread never dereferences the host's pointer.
    DXGI_OUTPUT_DESC1 override_desc;
    bool have_override_desc;
    _Atomic bool registered;

    // Per ring slot: the generation of the frame the host is holding in it,
    // or 0 when the slot is free. A release names the generation it releases
    // and takes the slot with a compare-exchange, so a release belonging to
    // a ring that has been superseded can never free a live slot. The
    // generation counter is on the core, not on the VO, so it keeps counting
    // across a VO that is destroyed and created again.
    _Atomic uint32_t slot_gen[MPV_SOCKPUPPET_D3D11_MAX_RING];
    _Atomic uint32_t generation;

    // The VO this context is running on, or NULL when there is none.
    //
    // Published by sp_init() and retracted by sp_uninit(), both on the VO
    // thread and both under vo_lock. mpv_sockpuppet_d3d11_wakeup() takes the
    // same lock around vo_redraw(), so a wakeup racing a teardown either gets
    // there first, and holds the lock the retraction is waiting for, or
    // arrives after it and finds NULL. It can never be handed a vo that is
    // being freed. This is the whole lifetime guarantee: the pointer is only
    // ever read under the lock, and it is never read anywhere else.
    //
    // vo_lock is a leaf. vo_redraw() takes vo_internal.lock beneath it, and
    // nothing that holds vo_internal.lock takes vo_lock: sp_init() and
    // sp_uninit() run from the VO thread's preinit and uninit, both outside
    // that lock. It is initialised by mp_clients_init() and destroyed by
    // mp_clients_destroy(), so it outlives every VO.
    mp_mutex vo_lock;
    struct vo *vo;

    // Set by mpv_sockpuppet_d3d11_wakeup() when it asks for a redraw, and
    // taken by the next sp_swap_buffers(), which presents that frame with a
    // sync interval of 0 instead of 1.
    //
    // An ordinary frame is presented on a vsync on purpose: DXGI pacing the
    // render loop to the display is what keeps the VO's frame timing, and
    // A/V sync with it, on a swapchain that has no window to pace it. A
    // redraw the host asked for is not an ordinary frame. It carries no new
    // video, it exists because something the host changed should be seen
    // before the next one, and every vsync slot it waits for is a slot the
    // film's own frames wanted. On a swapchain attached to no visual DXGI
    // paces against the adapter's output rather than the panel, so there are
    // about sixty slots a second to share however fast the display runs, and
    // a host animating a property at the display's rate loses two thirds of
    // its redraws to that queue while a film is playing. Paused, nothing
    // competes and the same animation is smooth, which is the whole shape of
    // the fault.
    //
    // The flag is written under vo_lock, beside the vo pointer and by the
    // same call, and read with an exchange from the VO thread, so a mark is
    // consumed exactly once and never twice. It is deliberately not paired
    // with the frame it was asked for: a redraw that happens to arrive at
    // the same moment as a decoded frame presents that one unsynced instead.
    // That costs one frame's pacing, once, and a mechanism to prevent it
    // would cost more than it saves.
    _Atomic bool redraw_pending;
};

extern const struct ra_ctx_fns ra_ctx_d3d11_sockpuppet;

// The two accessors gpu_next/context.c reaches through
// ra_d3d11_ctx_get_swapchain() and ra_d3d11_ctx_set_swapchain_params(), for
// this context. See context.c, which delegates here.
IDXGISwapChain *ra_d3d11_sockpuppet_get_swapchain(struct ra_ctx *ra);
struct pl_d3d11_swapchain_params;
void ra_d3d11_sockpuppet_set_swapchain_params(struct ra_ctx *ra,
                                              struct pl_d3d11_swapchain_params *params);
