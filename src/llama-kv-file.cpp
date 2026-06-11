#include "llama-kv-file.h"

#include "llama-impl.h"

#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

#if defined(_WIN32)
// File-backed KV cache is not supported on Windows in this patch.
ggml_backend_buffer_type_t llama_kv_file_buffer_type(const char * /*path*/) {
    return nullptr;
}
#else

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// KV tensors are 32-byte aligned in ggml; mmap() returns page-aligned memory
// which always satisfies this, so no realignment of the base is needed.
#define LLAMA_KV_FILE_ALIGNMENT 32

// per-buffer state
struct llama_kv_file_buffer {
    void *      addr = MAP_FAILED;
    size_t      size = 0;
    int         fd   = -1;
    std::string path;
};

// per-buffer-type state (carries the backing path)
struct llama_kv_file_buft {
    std::string base_path;
    int         n_live = 0; // number of currently-live buffers from this buft
};

static void * llama_kv_file_get_base(ggml_backend_buffer_t buffer) {
    auto * b = (llama_kv_file_buffer *) buffer->context;
    return b->addr;
}

static void llama_kv_file_free_buffer(ggml_backend_buffer_t buffer) {
    auto * b = (llama_kv_file_buffer *) buffer->context;
    if (b->addr && b->addr != MAP_FAILED) {
        munmap(b->addr, b->size);
    }
    if (b->fd >= 0) {
        close(b->fd);
    }
    if (buffer->buft && buffer->buft->context) {
        auto * bt = (llama_kv_file_buft *) buffer->buft->context;
        if (bt->n_live > 0) {
            bt->n_live--; // allow the next (sequential) context to reuse the same file
        }
    }
    delete b;
}

static void llama_kv_file_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void llama_kv_file_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void llama_kv_file_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool llama_kv_file_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void llama_kv_file_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * b = (llama_kv_file_buffer *) buffer->context;
    memset(b->addr, value, b->size);
}

static const ggml_backend_buffer_i llama_kv_file_buffer_i = {
    /* .free_buffer   = */ llama_kv_file_free_buffer,
    /* .get_base      = */ llama_kv_file_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ llama_kv_file_memset_tensor,
    /* .set_tensor    = */ llama_kv_file_set_tensor,
    /* .get_tensor    = */ llama_kv_file_get_tensor,
    /* .cpy_tensor    = */ llama_kv_file_cpy_tensor,
    /* .clear         = */ llama_kv_file_clear,
    /* .reset         = */ NULL,
};

static const char * llama_kv_file_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_KV_File";
    GGML_UNUSED(buft);
}

static size_t llama_kv_file_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return LLAMA_KV_FILE_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool llama_kv_file_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t llama_kv_file_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * bt = (llama_kv_file_buft *) buft->context;

    // if a buft backs more than one concurrently-live buffer (e.g. iSWA's two
    // KV caches), give each its own file so the mappings do not overlap. The
    // index is freed again on buffer destruction so sequential contexts reuse
    // the same file rather than spawning cache.img.1, .2, ...
    const int idx = bt->n_live++;
    std::string path = bt->base_path;
    if (idx > 0) {
        path += "." + std::to_string(idx);
    }

    const int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        LLAMA_LOG_ERROR("%s: failed to open KV cache file '%s': %s\n", __func__, path.c_str(), strerror(errno));
        return nullptr;
    }

    // reserve the backing blocks up front (fallocate-style). The file may have
    // been pre-allocated by the caller; only grow it if it is too small.
    struct stat st;
    if (fstat(fd, &st) == 0 && (size_t) st.st_size < size) {
#if defined(__APPLE__)
        // posix_fallocate is unavailable on macOS; best-effort contiguous
        // preallocation via F_PREALLOCATE, then ensure the size with ftruncate.
        fstore_t fstore = { F_ALLOCATECONTIG | F_ALLOCATEALL, F_PEOFPOSMODE, 0, (off_t) size, 0 };
        if (fcntl(fd, F_PREALLOCATE, &fstore) == -1) {
            fstore.fst_flags = F_ALLOCATEALL;
            fcntl(fd, F_PREALLOCATE, &fstore);
        }
        if (ftruncate(fd, (off_t) size) != 0) {
            LLAMA_LOG_ERROR("%s: ftruncate('%s', %zu) failed: %s\n", __func__, path.c_str(), size, strerror(errno));
            close(fd);
            return nullptr;
        }
#else
        if (posix_fallocate(fd, 0, (off_t) size) != 0) {
            // fall back to a plain truncate (sparse) if the fs has no fallocate
            if (ftruncate(fd, (off_t) size) != 0) {
                LLAMA_LOG_ERROR("%s: posix_fallocate/ftruncate('%s', %zu) failed: %s\n", __func__, path.c_str(), size, strerror(errno));
                close(fd);
                return nullptr;
            }
        }
#endif
    }

    void * addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        LLAMA_LOG_ERROR("%s: mmap('%s', %zu) failed: %s\n", __func__, path.c_str(), size, strerror(errno));
        close(fd);
        return nullptr;
    }

    auto * b = new llama_kv_file_buffer{ addr, size, fd, path };

    LLAMA_LOG_INFO("%s: KV cache mmap'd to '%s' (%.2f MiB)\n", __func__, path.c_str(), size / 1024.0 / 1024.0);

    return ggml_backend_buffer_init(buft, llama_kv_file_buffer_i, b, size);
}

ggml_backend_buffer_type_t llama_kv_file_buffer_type(const char * path) {
    if (!path || !*path) {
        return nullptr;
    }

    // return the same buft per path so the KV cache groups all layers into a
    // single ggml context and a single backing file. Intentionally leaked for
    // the process lifetime (mirrors the static CPU buffer type).
    static std::map<std::string, ggml_backend_buffer_type *> registry;

    auto it = registry.find(path);
    if (it != registry.end()) {
        return it->second;
    }

    auto * buft = new ggml_backend_buffer_type{
        /* .iface = */ {
            /* .get_name         = */ llama_kv_file_buft_get_name,
            /* .alloc_buffer     = */ llama_kv_file_buft_alloc_buffer,
            /* .get_alignment    = */ llama_kv_file_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .is_host          = */ llama_kv_file_buft_is_host,
        },
        /* .device  = */ NULL, // host buffer; mirrors ggml_backend_cpu_buffer_type
        /* .context = */ new llama_kv_file_buft{ std::string(path), 0 },
    };

    registry[path] = buft;
    return buft;
}

#endif // _WIN32
