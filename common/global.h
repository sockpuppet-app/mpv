#ifndef MPV_MPV_H
#define MPV_MPV_H

// This should be accessed by glue code only, never normal code.
// The only purpose of this is to make mpv library-safe.
// Think hard before adding new members.
struct mpv_global {
    struct mp_log *log;
    struct m_config_shadow *config;
    struct mp_client_api *client_api;
    char *configdir;
    struct stats_base *stats;
    struct demux_packet_pool *packet_pool;
    struct curl_ctx *curl;
    // Set by mpv_sockpuppet_d3d11_set_host(); read by the sockpuppet-d3d11
    // render context. NULL unless a host registered.
    struct mp_sockpuppet_d3d11 *sockpuppet_d3d11;
};

#endif
