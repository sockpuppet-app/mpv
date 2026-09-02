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

#include "mpv/client.h"
#include "video/out/gpu/context.h"

// What mpv_sockpuppet_d3d11_set_host() registers on the mpv_global, and what
// the sockpuppet-d3d11 render context reads. One per core.
struct mp_sockpuppet_d3d11 {
    mpv_sockpuppet_d3d11_host host;
    // One flag per ring slot: 1 while the host holds the frame in it.
    _Atomic int busy[MPV_SOCKPUPPET_D3D11_MAX_RING];
};

extern const struct ra_ctx_fns ra_ctx_d3d11_sockpuppet;

// The two accessors gpu_next/context.c reaches through
// ra_d3d11_ctx_get_swapchain() and ra_d3d11_ctx_set_swapchain_params(), for
// this context. See context.c, which delegates here.
IDXGISwapChain *ra_d3d11_sockpuppet_get_swapchain(struct ra_ctx *ra);
struct pl_d3d11_swapchain_params;
void ra_d3d11_sockpuppet_set_swapchain_params(struct ra_ctx *ra,
                                              struct pl_d3d11_swapchain_params *params);
