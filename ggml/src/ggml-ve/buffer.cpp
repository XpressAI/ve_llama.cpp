#include "buffer.hpp"
#include "common.hpp"
#include "device.hpp"

#include "ggml-backend-impl.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace ggml_ve {

namespace {

struct hbm_buffer_state {
    VEDAdeviceptr hbm_ptr   = 0;
    size_t        size      = 0;
    int           device_id = 0;
};

// Helper: get the active primary context for a device. We use the cached
// `device::context` from device.cpp so we don't bump VEDA's primary-context
// retain count on every operation — the count is taken once during
// init_devices() and held for the lifetime of the process.
VEDAcontext active_ctx(int device_id) {
    device * d = device_at(device_id);
    return d ? d->context : nullptr;
}

void buf_free_buffer(ggml_backend_buffer_t buffer) {
    auto * st = (hbm_buffer_state *) buffer->context;
    if (st == nullptr) return;
    if (st->hbm_ptr != 0) {
        // Use the same context that was active for the alloc, and use the
        // async free + sync to match the async alloc (mixing sync/async
        // primitives can leak the underlying request slot in VEDA).
        VEDAContextGuard guard(active_ctx(st->device_id));
        if (guard.is_valid()) {
            vedaMemFreeAsync(st->hbm_ptr, 0);
            vedaCtxSynchronize();
        }
    }
    delete st;
}

void * buf_get_base(ggml_backend_buffer_t buffer) {
    auto * st = (hbm_buffer_state *) buffer->context;
    return reinterpret_cast<void *>((uintptr_t) st->hbm_ptr);
}

ggml_status buf_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

void buf_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                       uint8_t value, size_t offset, size_t size) {
    auto * st = (hbm_buffer_state *) buffer->context;
    VEDAContextGuard guard(active_ctx(st->device_id));
    if (!guard.is_valid()) return;

    VEDAdeviceptr p = (VEDAdeviceptr)(uintptr_t) tensor->data + offset;
    vedaMemsetD8Async(p, value, size, 0);
    vedaCtxSynchronize();
}

void buf_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                    const void * data, size_t offset, size_t size) {
    auto * st = (hbm_buffer_state *) buffer->context;
    VEDAContextGuard guard(active_ctx(st->device_id));
    if (!guard.is_valid()) return;

    VEDAdeviceptr p = (VEDAdeviceptr)(uintptr_t) tensor->data + offset;
    if (!ggml_ve_ok(vedaMemcpyHtoDAsync(p, data, size, 0),
                    "vedaMemcpyHtoDAsync (HBM set_tensor)")) {
        return;
    }
    vedaCtxSynchronize();
}

void buf_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                    void * data, size_t offset, size_t size) {
    auto * st = (hbm_buffer_state *) buffer->context;
    VEDAContextGuard guard(active_ctx(st->device_id));
    if (!guard.is_valid()) return;

    VEDAdeviceptr p = (VEDAdeviceptr)(uintptr_t) tensor->data + offset;
    if (!ggml_ve_ok(vedaMemcpyDtoHAsync(data, p, size, 0),
                    "vedaMemcpyDtoHAsync (HBM get_tensor)")) {
        return;
    }
    vedaCtxSynchronize();
}

void buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * st = (hbm_buffer_state *) buffer->context;
    VEDAContextGuard guard(active_ctx(st->device_id));
    if (!guard.is_valid()) return;
    vedaMemsetD8Async(st->hbm_ptr, value, st->size, 0);
    vedaCtxSynchronize();
}

const ggml_backend_buffer_i hbm_buffer_iface = {
    /* .free_buffer    = */ buf_free_buffer,
    /* .get_base       = */ buf_get_base,
    /* .init_tensor    = */ buf_init_tensor,
    /* .memset_tensor  = */ buf_memset_tensor,
    /* .set_tensor     = */ buf_set_tensor,
    /* .get_tensor     = */ buf_get_tensor,
    /* .set_tensor_2d  = */ nullptr,
    /* .get_tensor_2d  = */ nullptr,
    /* .cpy_tensor     = */ nullptr,
    /* .clear          = */ buf_clear,
    /* .reset          = */ nullptr,
};

// ---- Buffer type table ----
std::array<ggml_backend_buffer_type, GGML_VE_MAX_DEVICES> g_buft;
char                                                     g_buft_names[GGML_VE_MAX_DEVICES][16];
bool                                                     g_inited = false;

const char * buft_get_name(ggml_backend_buffer_type_t buft) {
    int device_id = (int)(intptr_t) buft->context;
    if (device_id < 0 || device_id >= GGML_VE_MAX_DEVICES) return "VE_HBM";
    return g_buft_names[device_id];
}

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    int device_id = (int)(intptr_t) buft->context;
    if (device_id < 0 || device_id >= device_count()) device_id = 0;

    const size_t page = 4096;
    size_t size_aligned = (size == 0) ? page : size;
    if ((size_aligned % page) != 0) {
        size_aligned += page - (size_aligned % page);
    }

    VEDAContextGuard guard(active_ctx(device_id));
    if (!guard.is_valid()) return nullptr;

    // Drain any in-flight async ops before allocating new HBM. Otherwise
    // back-to-back alloc/free runs hit VEDA_ERROR_INVALID_REQID because old
    // async ops still hold request slots.
    vedaCtxSynchronize();

    VEDAdeviceptr hbm_ptr = 0;
    if (!ggml_ve_ok(vedaMemAllocAsync(&hbm_ptr, size_aligned, 0),
                    "vedaMemAllocAsync (HBM alloc)")) {
        return nullptr;
    }
    vedaCtxSynchronize();

    // Zero-init: garbage in the KV cache causes wrong attention output.
    vedaMemsetD8Async(hbm_ptr, 0, size_aligned, 0);
    vedaCtxSynchronize();

    auto * st = new hbm_buffer_state{hbm_ptr, size_aligned, device_id};
    GGML_LOG_INFO("ggml-ve: allocated %.2f MiB HBM buffer on VE%d @0x%lx\n",
                  size_aligned / (1024.0 * 1024.0), device_id, (unsigned long) hbm_ptr);
    return ggml_backend_buffer_init(buft, hbm_buffer_iface, st, size_aligned);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 256;
}

bool buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

} // namespace

void buffer_types_init() {
    if (g_inited) return;
    init_devices_once();

    for (int i = 0; i < GGML_VE_MAX_DEVICES; ++i) {
        std::snprintf(g_buft_names[i], sizeof(g_buft_names[i]), "VE%d_HBM", i);
        g_buft[i] = ggml_backend_buffer_type{
            /* .iface   = */ {
                /* .get_name       = */ buft_get_name,
                /* .alloc_buffer   = */ buft_alloc_buffer,
                /* .get_alignment  = */ buft_get_alignment,
                /* .get_max_size   = */ nullptr,
                /* .get_alloc_size = */ nullptr,
                /* .is_host        = */ buft_is_host,
            },
            /* .device  = */ nullptr,  // ggml-ve.cpp populates this when registering devices
            /* .context = */ (void *)(intptr_t) i,
        };
    }
    g_inited = true;
}

ggml_backend_buffer_type_t hbm_buffer_type(int device_id) {
    buffer_types_init();
    if (device_id < 0 || device_id >= GGML_VE_MAX_DEVICES) device_id = 0;
    return &g_buft[device_id];
}

} // namespace ggml_ve
