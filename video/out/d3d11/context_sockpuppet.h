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
};

extern const struct ra_ctx_fns ra_ctx_d3d11_sockpuppet;

// The two accessors gpu_next/context.c reaches through
// ra_d3d11_ctx_get_swapchain() and ra_d3d11_ctx_set_swapchain_params(), for
// this context. See context.c, which delegates here.
IDXGISwapChain *ra_d3d11_sockpuppet_get_swapchain(struct ra_ctx *ra);
struct pl_d3d11_swapchain_params;
void ra_d3d11_sockpuppet_set_swapchain_params(struct ra_ctx *ra,
                                              struct pl_d3d11_swapchain_params *params);
