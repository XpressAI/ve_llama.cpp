#ifndef GGML_VE_HMEM_POOL_HPP
#define GGML_VE_HMEM_POOL_HPP

// HMEM (host-mapped memory) buffer pool. HMEM is host-resident memory the VE
// can read/write directly across PCIe. We recycle buffers across kernel
// launches to amortise allocation cost, and optionally remember which host
// pointer's contents were last copied into each buffer so a repeat call can
// skip the copy.

#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ggml_ve {

struct hmem_buffer {
    VEDAhmemptr   hmem        = 0;
    size_t        size        = 0;
    bool          in_use      = false;
    const void *  cached_data = nullptr;
    size_t        cached_size = 0;
};

class hmem_pool {
public:
    // Acquire a buffer >= size. If `data` is provided and the same pointer
    // (with the same size) is already cached in a free buffer, return that
    // buffer (caller skips the copy). Otherwise reuse a free buffer or alloc.
    VEDAhmemptr acquire(size_t size, const void * data = nullptr) {
        static const bool disable_weight_cache =
            (std::getenv("GGML_VE_NO_WEIGHT_CACHE") != nullptr);

        if (data != nullptr && !disable_weight_cache) {
            for (auto & buf : buffers_) {
                if (!buf.in_use && buf.cached_data == data && buf.cached_size == size) {
                    buf.in_use = true;
                    weight_hits_++;
                    return buf.hmem;
                }
            }
        }

        for (auto & buf : buffers_) {
            if (!buf.in_use && buf.size >= size) {
                buf.in_use = true;
                buf.cached_data = nullptr;
                buf.cached_size = 0;
                hits_++;
                return buf.hmem;
            }
        }

        VEDAhmemptr h = 0;
        VEDAresult err = vedaHMemAlloc(&h, size);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "hmem_pool: vedaHMemAlloc(size=%zu) failed: %s\n",
                    size, ggml_ve_err_str(err));
            return 0;
        }
        buffers_.push_back({h, size, true, nullptr, 0});
        total_allocated_ += size;
        misses_++;
        return h;
    }

    // Release back to pool. If `data` is given, tag the buffer with that
    // pointer so the next acquire(data) can return it directly. `clear_cache`
    // wipes that tag — used for in-place ops where the host pointer is alias
    // for both input and output.
    void release(VEDAhmemptr h,
                 const void * data = nullptr,
                 size_t data_size = 0,
                 bool clear_cache = false) {
        for (auto & buf : buffers_) {
            if (buf.hmem == h) {
                buf.in_use = false;
                if (clear_cache) {
                    buf.cached_data = nullptr;
                    buf.cached_size = 0;
                } else if (data != nullptr) {
                    buf.cached_data = data;
                    buf.cached_size = data_size;
                }
                return;
            }
        }
    }

    bool is_cached(const void * data, size_t size, VEDAhmemptr * out = nullptr) const {
        for (const auto & buf : buffers_) {
            if (buf.cached_data == data && buf.cached_size == size) {
                if (out) *out = buf.hmem;
                return true;
            }
        }
        return false;
    }

    void clear() {
        for (auto & buf : buffers_) {
            if (buf.hmem) vedaHMemFree(buf.hmem);
        }
        buffers_.clear();
        total_allocated_ = 0;
    }

    void stats(size_t * allocated, int64_t * hits, int64_t * misses, int64_t * whits) const {
        if (allocated) *allocated = total_allocated_;
        if (hits)      *hits      = hits_;
        if (misses)    *misses    = misses_;
        if (whits)     *whits     = weight_hits_;
    }

private:
    std::vector<hmem_buffer> buffers_;
    size_t  total_allocated_ = 0;
    int64_t hits_            = 0;
    int64_t misses_          = 0;
    int64_t weight_hits_     = 0;
};

} // namespace ggml_ve

#endif // GGML_VE_HMEM_POOL_HPP
