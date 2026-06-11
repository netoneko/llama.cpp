#pragma once

#include "ggml-backend.h"

// File-backed (mmap) KV-cache buffer type.
//
// Returns a host buffer type whose buffers are backed by an mmap'd, pre-sized
// file on disk instead of anonymous RAM. The mapping is MAP_SHARED with
// PROT_READ|PROT_WRITE and is_host == true, so all existing CPU read/write and
// attention paths work unchanged.
//
// Returns the same buft instance for a given path (so the KV cache groups all
// layers into a single context / single backing file). Returns nullptr if file
// mapping is not supported on this platform.
//
// See docs/LLAMA_CPP_MMAP_PATCH.md for the design rationale.
ggml_backend_buffer_type_t llama_kv_file_buffer_type(const char * path);
