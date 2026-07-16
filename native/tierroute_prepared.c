/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Project-owned, dependency-free C11 prepared-session sidecar.
 *
 * The wire contract is frozen in docs/native-prepared-session-protocol.md.
 * This is deliberately a separate executable and protocol from TRRIDG01.  The
 * complete canonical nested-LODO graph is admitted before large allocation,
 * then executed in one process.  Feature rows remain file-backed: only one row
 * is present on the C heap while reusable per-domain moments are accumulated.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#define SESSION_HEADER_BYTES ((size_t)160)
#define STORE_HEADER_BYTES ((size_t)472)
#define RESULT_HEADER_BYTES ((size_t)448)
#define DIGEST_BYTES ((size_t)32)
#define MAX_DOMAINS ((size_t)7)
#define MAX_SUBSETS ((size_t)63)
#define MAX_BLOCKS ((size_t)154)
#define UNIVERSAL_SURFACE_WIDTH UINT64_C(12)
#define CONTINUOUS_WIDTH ((size_t)3)
#define TAG_OFFSET ((size_t)5)
#define TAG_COUNT ((size_t)7)
#define MAX_ROW_ID_BYTES ((size_t)4096)
#define MAX_STORE_BYTES UINT64_C(536870912)
#define MAX_RESULT_BYTES UINT64_C(134217728)
#define MAX_HEAP_BYTES UINT64_C(536870912)
#define MAX_PRIVATE_SCRATCH_BYTES UINT64_C(1073741824)
#define MAX_WORK_UNITS UINT64_C(200000000000)
#define MAX_ROWS UINT64_C(1000000)
#define MAX_FEATURES UINT64_C(4096)
#define MAX_TARGETS UINT64_C(256)

enum prepared_status {
    PREPARED_SUCCESS = 0,
    PREPARED_PROTOCOL_ERROR = 1,
    PREPARED_RESOURCE_ERROR = 2,
    PREPARED_NUMERIC_ERROR = 3,
    PREPARED_ALLOCATION_ERROR = 4,
    PREPARED_SOLVE_ERROR = 5,
    PREPARED_INTERNAL_ERROR = 6
};

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t used;
};

struct session_request {
    unsigned char nonce[DIGEST_BYTES];
    unsigned char store_sha256[DIGEST_BYTES];
    unsigned char binary_sha256[DIGEST_BYTES];
    uint64_t request_bytes;
    uint64_t expected_result_bytes;
    double ridge;
    int parsed;
};

struct prepared_store {
    uint64_t file_bytes;
    uint64_t domain_count;
    uint64_t row_count;
    uint64_t feature_count;
    uint64_t target_count;
    uint64_t row_key_offset;
    uint64_t row_key_bytes;
    uint64_t domain_offset;
    uint64_t domain_bytes;
    uint64_t feature_offset;
    uint64_t feature_bytes;
    uint64_t target_offset;
    uint64_t target_bytes;
    unsigned char graph_sha256[DIGEST_BYTES];
    unsigned char source_fit_sha256[DIGEST_BYTES];
    unsigned char logical_store_sha256[DIGEST_BYTES];
    unsigned char embedding_snapshot_sha256[DIGEST_BYTES];
    unsigned char embedding_identity_sha256[DIGEST_BYTES];
    unsigned char model_catalogue_sha256[DIGEST_BYTES];
    unsigned char payload_sha256[DIGEST_BYTES];
    uint64_t domain_counts[MAX_DOMAINS];
    uint64_t domain_masks[MAX_DOMAINS];
    int parsed;
};

struct prepared_plan {
    size_t domain_count;
    size_t row_count;
    size_t feature_count;
    size_t target_count;
    uint64_t domain_rows[MAX_DOMAINS];
    size_t packed_count;
    size_t target_cells;
    size_t domain_moment_cells;
    size_t subset_count;
    size_t block_count;
    uint64_t subset_masks[MAX_SUBSETS];
    uint64_t subset_rows[MAX_SUBSETS];
    uint64_t subset_tag_masks[MAX_SUBSETS];
    size_t active_widths[MAX_SUBSETS];
    size_t weight_offsets[MAX_SUBSETS];
    size_t block_subset[MAX_BLOCKS];
    size_t block_domain[MAX_BLOCKS];
    uint64_t block_rows[MAX_BLOCKS];
    size_t score_offsets[MAX_BLOCKS];
    int block_lookup[MAX_SUBSETS][MAX_DOMAINS];
    size_t weight_cells;
    size_t score_cells;
    size_t maximum_width;
    uint64_t score_memberships;
    uint64_t coefficient_section_bytes;
    uint64_t score_section_bytes;
    uint64_t result_bytes;
    uint64_t statistics_work;
    uint64_t solve_work;
    uint64_t score_work;
    uint64_t modeled_heap_bytes;
    uint64_t input_validation_bytes;
    uint64_t output_numeric_cells;
    uint64_t file_backed_input_bytes;
    int admitted;
};

struct prepared_buffers {
    unsigned char *domain_indices;
    double *targets;
    double *domain_moments;
    double *coefficient_means;
    double *coefficient_scales;
    double *intercepts;
    double *weights;
    double *scores;
    double *feature_row;
    double *delta_x;
    double *delta_y;
    double *subset_x_mean;
    double *subset_y_mean;
    double *subset_xx;
    double *subset_xy;
    double *gram;
    double *cholesky;
    double *right_hand_sides;
};

static const unsigned char SESSION_MAGIC[8] =
    {'T', 'R', 'P', 'S', 'E', 'S', '0', '1'};
static const unsigned char STORE_MAGIC[8] =
    {'T', 'R', 'P', 'S', 'T', 'O', '0', '1'};
static const unsigned char RESULT_MAGIC[8] =
    {'T', 'R', 'P', 'R', 'E', 'S', '0', '1'};
static int checked_add_u64(uint64_t first, uint64_t second, uint64_t *result)
{
    if (first > UINT64_MAX - second) {
        return 0;
    }
    *result = first + second;
    return 1;
}

static int checked_multiply_u64(uint64_t first, uint64_t second, uint64_t *result)
{
    if (first != 0U && second > UINT64_MAX / first) {
        return 0;
    }
    *result = first * second;
    return 1;
}

static int checked_accumulate_u64(uint64_t *total, uint64_t value)
{
    uint64_t updated;
    if (!checked_add_u64(*total, value, &updated)) {
        return 0;
    }
    *total = updated;
    return 1;
}

static int checked_accumulate_product_u64(
    uint64_t *total,
    uint64_t first,
    uint64_t second
)
{
    uint64_t product;
    return checked_multiply_u64(first, second, &product) &&
           checked_accumulate_u64(total, product);
}

static uint32_t decode_u32(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0]) |
           ((uint32_t)bytes[1] << 8U) |
           ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

static uint64_t decode_u64(const unsigned char *bytes)
{
    uint64_t value = 0U;
    size_t index;
    for (index = 0U; index < 8U; ++index) {
        value |= ((uint64_t)bytes[index]) << (8U * index);
    }
    return value;
}

static void encode_u32(unsigned char *bytes, uint32_t value)
{
    size_t index;
    for (index = 0U; index < 4U; ++index) {
        bytes[index] = (unsigned char)((value >> (8U * index)) & UINT32_C(255));
    }
}

static void encode_u64(unsigned char *bytes, uint64_t value)
{
    size_t index;
    for (index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)((value >> (8U * index)) & UINT64_C(255));
    }
}

static int all_zero(const unsigned char *bytes, size_t count)
{
    size_t index;
    unsigned char combined = 0U;
    for (index = 0U; index < count; ++index) {
        combined = (unsigned char)(combined | bytes[index]);
    }
    return combined == 0U;
}

static int platform_is_supported(void)
{
    const uint16_t integer_one = UINT16_C(1);
    const double floating_one = 1.0;
    unsigned char integer_bytes[sizeof(integer_one)];
    uint64_t floating_bits = 0U;
    if (CHAR_BIT != 8 || sizeof(uint32_t) != 4U || sizeof(uint64_t) != 8U ||
        sizeof(double) != 8U || FLT_RADIX != 2 || DBL_MANT_DIG != 53 ||
        DBL_MAX_EXP != 1024) {
        return 0;
    }
    memcpy(integer_bytes, &integer_one, sizeof(integer_one));
    memcpy(&floating_bits, &floating_one, sizeof(floating_one));
    return integer_bytes[0] == 1U && floating_bits == UINT64_C(0x3ff0000000000000);
}

static int configure_binary_streams(void)
{
#if defined(_WIN32)
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1) {
        return 0;
    }
#endif
    return 1;
}

static enum prepared_status read_exact(unsigned char *destination, size_t count)
{
    size_t offset = 0U;
    while (offset < count) {
        const size_t received = fread(destination + offset, 1U, count - offset, stdin);
        if (received > 0U) {
            offset += received;
        } else if (ferror(stdin)) {
            return PREPARED_INTERNAL_ERROR;
        } else {
            return PREPARED_PROTOCOL_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static int seek_input(uint64_t offset)
{
    if (offset > (uint64_t)LONG_MAX) {
        return 0;
    }
    clearerr(stdin);
    return fseek(stdin, (long)offset, SEEK_SET) == 0;
}

static int canonical_f64(double value)
{
    uint64_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return isfinite(value) && !(value == 0.0 && (bits >> 63U) != 0U);
}

static double positive_zero(double value)
{
    return value == 0.0 ? 0.0 : value;
}

/* Small project-owned SHA-256 used only to validate the already-authenticated input. */
static uint32_t rotate_right(uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32U - count));
}

static void sha256_transform(struct sha256_context *context, const unsigned char *block)
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
        UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
        UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
        UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
        UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
        UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
        UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
        UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
        UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
        UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
        UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
        UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;
    for (index = 0U; index < 16U; ++index) {
        const size_t offset = index * 4U;
        words[index] = ((uint32_t)block[offset] << 24U) |
                       ((uint32_t)block[offset + 1U] << 16U) |
                       ((uint32_t)block[offset + 2U] << 8U) |
                       (uint32_t)block[offset + 3U];
    }
    for (index = 16U; index < 64U; ++index) {
        const uint32_t first = rotate_right(words[index - 15U], 7U) ^
                               rotate_right(words[index - 15U], 18U) ^
                               (words[index - 15U] >> 3U);
        const uint32_t second = rotate_right(words[index - 2U], 17U) ^
                                rotate_right(words[index - 2U], 19U) ^
                                (words[index - 2U] >> 10U);
        words[index] = words[index - 16U] + first + words[index - 7U] + second;
    }
    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0U; index < 64U; ++index) {
        const uint32_t sum_one = h +
            (rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U)) +
            ((e & f) ^ ((~e) & g)) + constants[index] + words[index];
        const uint32_t sum_zero =
            (rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U)) +
            ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + sum_one;
        d = c;
        c = b;
        b = a;
        a = sum_one + sum_zero;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(struct sha256_context *context)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85), UINT32_C(0x3c6ef372),
        UINT32_C(0xa54ff53a), UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(context->state, initial, sizeof(initial));
    context->bit_count = 0U;
    context->used = 0U;
    memset(context->block, 0, sizeof(context->block));
}

static int sha256_update(
    struct sha256_context *context,
    const unsigned char *bytes,
    size_t count
)
{
    size_t offset = 0U;
    if (count > (size_t)((UINT64_MAX - context->bit_count) / UINT64_C(8))) {
        return 0;
    }
    context->bit_count += (uint64_t)count * UINT64_C(8);
    while (offset < count) {
        const size_t available = sizeof(context->block) - context->used;
        const size_t copied = count - offset < available ? count - offset : available;
        memcpy(context->block + context->used, bytes + offset, copied);
        context->used += copied;
        offset += copied;
        if (context->used == sizeof(context->block)) {
            sha256_transform(context, context->block);
            context->used = 0U;
        }
    }
    return 1;
}

static void sha256_final(struct sha256_context *context, unsigned char digest[DIGEST_BYTES])
{
    size_t index;
    context->block[context->used++] = 0x80U;
    if (context->used > 56U) {
        while (context->used < 64U) {
            context->block[context->used++] = 0U;
        }
        sha256_transform(context, context->block);
        context->used = 0U;
    }
    while (context->used < 56U) {
        context->block[context->used++] = 0U;
    }
    for (index = 0U; index < 8U; ++index) {
        context->block[56U + index] =
            (unsigned char)((context->bit_count >> (56U - 8U * index)) & UINT64_C(255));
    }
    sha256_transform(context, context->block);
    for (index = 0U; index < 8U; ++index) {
        digest[index * 4U] = (unsigned char)(context->state[index] >> 24U);
        digest[index * 4U + 1U] = (unsigned char)(context->state[index] >> 16U);
        digest[index * 4U + 2U] = (unsigned char)(context->state[index] >> 8U);
        digest[index * 4U + 3U] = (unsigned char)context->state[index];
    }
}

static enum prepared_status read_session_header(struct session_request *request)
{
    unsigned char header[SESSION_HEADER_BYTES];
    uint64_t ridge_bits;
    enum prepared_status status = read_exact(header, sizeof(header));
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    memcpy(request->nonce, header + 16U, DIGEST_BYTES);
    memcpy(request->store_sha256, header + 48U, DIGEST_BYTES);
    memcpy(request->binary_sha256, header + 80U, DIGEST_BYTES);
    request->request_bytes = decode_u64(header + 112U);
    request->expected_result_bytes = decode_u64(header + 120U);
    ridge_bits = decode_u64(header + 128U);
    memcpy(&request->ridge, &ridge_bits, sizeof(request->ridge));
    request->parsed = 1;
    if (memcmp(header, SESSION_MAGIC, sizeof(SESSION_MAGIC)) != 0 ||
        decode_u32(header + 8U) != UINT32_C(1) ||
        decode_u32(header + 12U) != UINT32_C(0) ||
        !all_zero(header + 136U, 24U) ||
        !canonical_f64(request->ridge) || !(request->ridge > 0.0)) {
        return PREPARED_PROTOCOL_ERROR;
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status read_store_header(struct prepared_store *store)
{
    unsigned char header[STORE_HEADER_BYTES];
    size_t index;
    enum prepared_status status = read_exact(header, sizeof(header));
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    store->file_bytes = decode_u64(header + 24U);
    store->domain_count = decode_u64(header + 32U);
    store->row_count = decode_u64(header + 40U);
    store->feature_count = decode_u64(header + 48U);
    store->target_count = decode_u64(header + 56U);
    store->row_key_offset = decode_u64(header + 72U);
    store->row_key_bytes = decode_u64(header + 80U);
    store->domain_offset = decode_u64(header + 88U);
    store->domain_bytes = decode_u64(header + 96U);
    store->feature_offset = decode_u64(header + 104U);
    store->feature_bytes = decode_u64(header + 112U);
    store->target_offset = decode_u64(header + 120U);
    store->target_bytes = decode_u64(header + 128U);
    memcpy(store->graph_sha256, header + 136U, DIGEST_BYTES);
    memcpy(store->source_fit_sha256, header + 168U, DIGEST_BYTES);
    memcpy(store->logical_store_sha256, header + 200U, DIGEST_BYTES);
    memcpy(store->embedding_snapshot_sha256, header + 232U, DIGEST_BYTES);
    memcpy(store->embedding_identity_sha256, header + 264U, DIGEST_BYTES);
    memcpy(store->model_catalogue_sha256, header + 296U, DIGEST_BYTES);
    memcpy(store->payload_sha256, header + 328U, DIGEST_BYTES);
    for (index = 0U; index < MAX_DOMAINS; ++index) {
        store->domain_counts[index] = decode_u64(header + 360U + 8U * index);
        store->domain_masks[index] = decode_u64(header + 416U + 8U * index);
    }
    store->parsed = 1;
    if (memcmp(header, STORE_MAGIC, sizeof(STORE_MAGIC)) != 0 ||
        decode_u32(header + 8U) != UINT32_C(1) ||
        decode_u32(header + 12U) != UINT32_C(0) ||
        decode_u64(header + 16U) != (uint64_t)STORE_HEADER_BYTES ||
        decode_u64(header + 64U) != UNIVERSAL_SURFACE_WIDTH) {
        return PREPARED_PROTOCOL_ERROR;
    }
    return PREPARED_SUCCESS;
}

static unsigned int popcount_u64(uint64_t value)
{
    unsigned int count = 0U;
    while (value != 0U) {
        count += (unsigned int)(value & UINT64_C(1));
        value >>= 1U;
    }
    return count;
}

static int append_subset(
    struct prepared_plan *plan,
    const struct prepared_store *store,
    uint64_t omitted_mask
)
{
    const uint64_t full_mask = (UINT64_C(1) << store->domain_count) - UINT64_C(1);
    const uint64_t training_mask = full_mask ^ omitted_mask;
    uint64_t rows = 0U;
    uint64_t tags = 0U;
    size_t domain;
    size_t width;
    if (plan->subset_count >= MAX_SUBSETS) {
        return 0;
    }
    for (domain = 0U; domain < plan->domain_count; ++domain) {
        if ((training_mask & (UINT64_C(1) << domain)) != 0U) {
            if (!checked_accumulate_u64(&rows, store->domain_counts[domain])) {
                return 0;
            }
            tags |= store->domain_masks[domain];
        }
    }
    width = TAG_OFFSET + (size_t)popcount_u64(tags) +
            (plan->feature_count - (size_t)UNIVERSAL_SURFACE_WIDTH);
    if (rows == 0U || width == 0U || width > plan->feature_count) {
        return 0;
    }
    plan->subset_masks[plan->subset_count] = training_mask;
    plan->subset_rows[plan->subset_count] = rows;
    plan->subset_tag_masks[plan->subset_count] = tags;
    plan->active_widths[plan->subset_count] = width;
    if (width > plan->maximum_width) {
        plan->maximum_width = width;
    }
    ++plan->subset_count;
    return 1;
}

static int enumerate_graph(struct prepared_plan *plan, const struct prepared_store *store)
{
    size_t first;
    size_t second;
    size_t third;
    size_t subset;
    size_t domain;
    for (first = 0U; first + 2U < plan->domain_count; ++first) {
        for (second = first + 1U; second + 1U < plan->domain_count; ++second) {
            for (third = second + 1U; third < plan->domain_count; ++third) {
                const uint64_t omitted = (UINT64_C(1) << first) |
                                         (UINT64_C(1) << second) |
                                         (UINT64_C(1) << third);
                if (!append_subset(plan, store, omitted)) {
                    return 0;
                }
            }
        }
    }
    for (first = 0U; first + 1U < plan->domain_count; ++first) {
        for (second = first + 1U; second < plan->domain_count; ++second) {
            const uint64_t omitted = (UINT64_C(1) << first) |
                                     (UINT64_C(1) << second);
            if (!append_subset(plan, store, omitted)) {
                return 0;
            }
        }
    }
    for (first = 0U; first < plan->domain_count; ++first) {
        if (!append_subset(plan, store, UINT64_C(1) << first)) {
            return 0;
        }
    }
    for (subset = 0U; subset < MAX_SUBSETS; ++subset) {
        for (domain = 0U; domain < MAX_DOMAINS; ++domain) {
            plan->block_lookup[subset][domain] = -1;
        }
    }
    for (subset = 0U; subset < plan->subset_count; ++subset) {
        for (domain = 0U; domain < plan->domain_count; ++domain) {
            if ((plan->subset_masks[subset] & (UINT64_C(1) << domain)) == 0U) {
                if (plan->block_count >= MAX_BLOCKS) {
                    return 0;
                }
                plan->block_subset[plan->block_count] = subset;
                plan->block_domain[plan->block_count] = domain;
                plan->block_rows[plan->block_count] = store->domain_counts[domain];
                plan->block_lookup[subset][domain] = (int)plan->block_count;
                ++plan->block_count;
            }
        }
    }
    return 1;
}

static int add_cubic_work(uint64_t *total, uint64_t width)
{
    uint64_t square;
    uint64_t cube;
    return checked_multiply_u64(width, width, &square) &&
           checked_multiply_u64(square, width, &cube) &&
           checked_accumulate_u64(total, cube);
}

static enum prepared_status preflight_session(
    const struct session_request *request,
    const struct prepared_store *store,
    struct prepared_plan *plan
)
{
    uint64_t expected;
    uint64_t feature_cells;
    uint64_t target_cells;
    uint64_t packed_twice;
    uint64_t packed;
    uint64_t domain_moment_cells;
    uint64_t sum_counts = 0U;
    uint64_t coefficient_bytes = 0U;
    uint64_t coefficient_numeric_cells = 0U;
    uint64_t score_bytes = 0U;
    uint64_t score_cells = 0U;
    uint64_t weight_cells = 0U;
    uint64_t numeric_cells = 0U;
    uint64_t heap_bytes;
    uint64_t scratch_bytes;
    uint64_t domain_end;
    uint64_t work_term;
    uint64_t maximum_square;
    size_t index;
    if (store->domain_count < UINT64_C(4) || store->domain_count > UINT64_C(7) ||
        store->row_count == 0U || store->row_count > MAX_ROWS ||
        store->feature_count < UNIVERSAL_SURFACE_WIDTH ||
        store->feature_count > MAX_FEATURES ||
        store->target_count == 0U || store->target_count > MAX_TARGETS ||
        store->file_bytes < (uint64_t)STORE_HEADER_BYTES ||
        store->file_bytes > MAX_STORE_BYTES) {
        return PREPARED_RESOURCE_ERROR;
    }
    if (store->row_key_offset != (uint64_t)STORE_HEADER_BYTES ||
        !checked_add_u64(store->row_key_offset, store->row_key_bytes, &expected) ||
        store->domain_offset != expected || store->domain_bytes != store->row_count ||
        !checked_add_u64(store->domain_offset, store->domain_bytes, &expected)) {
        return PREPARED_PROTOCOL_ERROR;
    }
    domain_end = expected;
    if (!checked_add_u64(expected, UINT64_C(7), &expected)) {
        return PREPARED_RESOURCE_ERROR;
    }
    expected &= ~UINT64_C(7);
    if (store->feature_offset != expected ||
        !checked_multiply_u64(store->row_count, store->feature_count, &feature_cells) ||
        !checked_multiply_u64(feature_cells, UINT64_C(8), &expected) ||
        store->feature_bytes != expected ||
        !checked_add_u64(store->feature_offset, store->feature_bytes, &expected) ||
        store->target_offset != expected ||
        !checked_multiply_u64(store->row_count, store->target_count, &target_cells) ||
        !checked_multiply_u64(target_cells, UINT64_C(8), &expected) ||
        store->target_bytes != expected ||
        !checked_add_u64(store->target_offset, store->target_bytes, &expected) ||
        store->file_bytes != expected) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if (!checked_add_u64((uint64_t)SESSION_HEADER_BYTES, store->file_bytes, &expected) ||
        request->request_bytes != expected ||
        request->expected_result_bytes < (uint64_t)RESULT_HEADER_BYTES ||
        request->expected_result_bytes > MAX_RESULT_BYTES) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if (!checked_multiply_u64(store->row_count, UINT64_C(35), &expected) ||
        store->row_key_bytes < expected ||
        !checked_multiply_u64(store->row_count, UINT64_C(4130), &expected) ||
        store->row_key_bytes > expected) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if (all_zero(request->nonce, DIGEST_BYTES) ||
        all_zero(request->store_sha256, DIGEST_BYTES) ||
        all_zero(request->binary_sha256, DIGEST_BYTES) ||
        all_zero(store->graph_sha256, DIGEST_BYTES) ||
        all_zero(store->source_fit_sha256, DIGEST_BYTES) ||
        all_zero(store->logical_store_sha256, DIGEST_BYTES) ||
        all_zero(store->model_catalogue_sha256, DIGEST_BYTES) ||
        all_zero(store->payload_sha256, DIGEST_BYTES)) {
        return PREPARED_PROTOCOL_ERROR;
    }
    for (index = 0U; index < MAX_DOMAINS; ++index) {
        if (index < (size_t)store->domain_count) {
            if (store->domain_counts[index] == 0U ||
                store->domain_counts[index] > store->row_count ||
                store->domain_masks[index] >= (UINT64_C(1) << TAG_COUNT) ||
                !checked_accumulate_u64(&sum_counts, store->domain_counts[index])) {
                return PREPARED_PROTOCOL_ERROR;
            }
        } else if (store->domain_counts[index] != 0U || store->domain_masks[index] != 0U) {
            return PREPARED_PROTOCOL_ERROR;
        }
    }
    if (sum_counts != store->row_count) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if ((store->feature_count == UNIVERSAL_SURFACE_WIDTH &&
         (!all_zero(store->embedding_snapshot_sha256, DIGEST_BYTES) ||
          !all_zero(store->embedding_identity_sha256, DIGEST_BYTES))) ||
        (store->feature_count > UNIVERSAL_SURFACE_WIDTH &&
         (all_zero(store->embedding_snapshot_sha256, DIGEST_BYTES) ||
          all_zero(store->embedding_identity_sha256, DIGEST_BYTES)))) {
        return PREPARED_PROTOCOL_ERROR;
    }
    plan->domain_count = (size_t)store->domain_count;
    plan->row_count = (size_t)store->row_count;
    plan->feature_count = (size_t)store->feature_count;
    plan->target_count = (size_t)store->target_count;
    for (index = 0U; index < plan->domain_count; ++index) {
        plan->domain_rows[index] = store->domain_counts[index];
    }
    if (!checked_add_u64(store->feature_count, UINT64_C(1), &packed_twice) ||
        !checked_multiply_u64(store->feature_count, packed_twice, &packed_twice)) {
        return PREPARED_RESOURCE_ERROR;
    }
    packed = packed_twice / UINT64_C(2);
    if (!checked_add_u64(store->feature_count, store->target_count, &domain_moment_cells) ||
        !checked_accumulate_u64(&domain_moment_cells, packed) ||
        !checked_accumulate_product_u64(
            &domain_moment_cells,
            store->feature_count,
            store->target_count
        )) {
        return PREPARED_RESOURCE_ERROR;
    }
    if (packed > (uint64_t)SIZE_MAX || target_cells > (uint64_t)SIZE_MAX ||
        domain_moment_cells > (uint64_t)SIZE_MAX) {
        return PREPARED_RESOURCE_ERROR;
    }
    plan->packed_count = (size_t)packed;
    plan->target_cells = (size_t)target_cells;
    plan->domain_moment_cells = (size_t)domain_moment_cells;
    if (!enumerate_graph(plan, store)) {
        return PREPARED_RESOURCE_ERROR;
    }
    for (index = 0U; index < plan->subset_count; ++index) {
        const uint64_t width = (uint64_t)plan->active_widths[index];
        uint64_t payload_cells = UINT64_C(6) + store->target_count;
        uint64_t payload_bytes;
        uint64_t square;
        if (weight_cells > (uint64_t)SIZE_MAX) {
            return PREPARED_RESOURCE_ERROR;
        }
        plan->weight_offsets[index] = (size_t)weight_cells;
        if (!checked_accumulate_product_u64(&payload_cells, store->target_count, width) ||
            !checked_accumulate_u64(&coefficient_numeric_cells, payload_cells) ||
            !checked_multiply_u64(payload_cells, UINT64_C(8), &payload_bytes) ||
            !checked_accumulate_u64(&coefficient_bytes, UINT64_C(48)) ||
            !checked_accumulate_u64(&coefficient_bytes, payload_bytes) ||
            !checked_accumulate_product_u64(&weight_cells, store->target_count, width) ||
            !add_cubic_work(&plan->solve_work, width) ||
            !checked_multiply_u64(width, width, &square) ||
            !checked_accumulate_product_u64(
                &plan->solve_work,
                UINT64_C(2) * store->target_count,
                square
            ) ||
            !checked_accumulate_product_u64(
                &plan->solve_work,
                store->target_count,
                width
            )) {
            return PREPARED_RESOURCE_ERROR;
        }
    }
    for (index = 0U; index < plan->block_count; ++index) {
        const uint64_t rows = plan->block_rows[index];
        const uint64_t width = (uint64_t)plan->active_widths[plan->block_subset[index]];
        uint64_t cells;
        uint64_t bytes;
        if (score_cells > (uint64_t)SIZE_MAX) {
            return PREPARED_RESOURCE_ERROR;
        }
        plan->score_offsets[index] = (size_t)score_cells;
        if (!checked_multiply_u64(rows, store->target_count, &cells) ||
            !checked_multiply_u64(cells, UINT64_C(8), &bytes) ||
            !checked_accumulate_u64(&score_bytes, UINT64_C(32)) ||
            !checked_accumulate_u64(&score_bytes, bytes) ||
            !checked_accumulate_u64(&score_cells, cells) ||
            !checked_accumulate_u64(&plan->score_memberships, rows) ||
            !checked_accumulate_product_u64(&plan->score_work, cells, width)) {
            return PREPARED_RESOURCE_ERROR;
        }
    }
    if (weight_cells > (uint64_t)SIZE_MAX || score_cells > (uint64_t)SIZE_MAX) {
        return PREPARED_RESOURCE_ERROR;
    }
    plan->weight_cells = (size_t)weight_cells;
    plan->score_cells = (size_t)score_cells;
    plan->coefficient_section_bytes = coefficient_bytes;
    plan->score_section_bytes = score_bytes;
    if (!checked_add_u64((uint64_t)RESULT_HEADER_BYTES, coefficient_bytes, &plan->result_bytes) ||
        !checked_accumulate_u64(&plan->result_bytes, score_bytes) ||
        plan->result_bytes != request->expected_result_bytes ||
        plan->result_bytes > MAX_RESULT_BYTES) {
        return PREPARED_RESOURCE_ERROR;
    }
    /* Frozen graph work units; these intentionally match the Python preflight. */
    if (!checked_add_u64(store->feature_count, store->target_count, &work_term) ||
        !checked_multiply_u64(work_term, UINT64_C(3), &work_term) ||
        !checked_accumulate_product_u64(&work_term, UINT64_C(1), packed) ||
        !checked_accumulate_product_u64(
            &work_term,
            store->feature_count,
            store->target_count
        ) ||
        !checked_multiply_u64(store->row_count, work_term, &plan->statistics_work)) {
        return PREPARED_RESOURCE_ERROR;
    }
    if (!checked_add_u64(plan->statistics_work, plan->solve_work, &work_term) ||
        !checked_accumulate_u64(&work_term, plan->score_work) ||
        work_term > MAX_WORK_UNITS) {
        return PREPARED_RESOURCE_ERROR;
    }
    /*
     * Conservative C-heap model: retained target/domain-moment/coefficient/score
     * cells plus the largest reusable subset workspace and one feature row. The
     * 72 KiB allowance covers row-key buffers and allocator bookkeeping.
     */
    if (!checked_accumulate_u64(&numeric_cells, target_cells) ||
        !checked_accumulate_product_u64(
            &numeric_cells,
            store->domain_count,
            domain_moment_cells
        ) ||
        !checked_accumulate_product_u64(
            &numeric_cells,
            (uint64_t)plan->subset_count,
            UINT64_C(6) + store->target_count
        ) ||
        !checked_accumulate_u64(&numeric_cells, weight_cells) ||
        !checked_accumulate_u64(&numeric_cells, score_cells) ||
        !checked_accumulate_product_u64(&numeric_cells, UINT64_C(3), store->feature_count) ||
        !checked_accumulate_product_u64(&numeric_cells, UINT64_C(2), store->target_count) ||
        !checked_accumulate_u64(&numeric_cells, packed) ||
        !checked_accumulate_product_u64(
            &numeric_cells,
            store->feature_count,
            store->target_count
        ) ||
        !checked_multiply_u64(
            (uint64_t)plan->maximum_width,
            (uint64_t)plan->maximum_width,
            &maximum_square
        ) ||
        !checked_accumulate_product_u64(&numeric_cells, UINT64_C(2), maximum_square) ||
        !checked_accumulate_product_u64(
            &numeric_cells,
            store->target_count,
            (uint64_t)plan->maximum_width
        ) ||
        !checked_multiply_u64(numeric_cells, UINT64_C(8), &heap_bytes) ||
        !checked_accumulate_u64(&heap_bytes, store->row_count) ||
        !checked_accumulate_u64(&heap_bytes, UINT64_C(73728)) ||
        heap_bytes > MAX_HEAP_BYTES) {
        return PREPARED_RESOURCE_ERROR;
    }
    plan->modeled_heap_bytes = heap_bytes;
    if (!checked_add_u64(request->request_bytes, plan->result_bytes, &scratch_bytes) ||
        scratch_bytes > MAX_PRIVATE_SCRATCH_BYTES) {
        return PREPARED_RESOURCE_ERROR;
    }
    if (!checked_add_u64(coefficient_numeric_cells, score_cells, &plan->output_numeric_cells) ||
        !checked_add_u64(store->file_bytes, store->row_key_bytes, &plan->input_validation_bytes) ||
        !checked_accumulate_u64(&plan->input_validation_bytes, store->domain_bytes) ||
        !checked_accumulate_u64(
            &plan->input_validation_bytes,
            store->feature_offset - domain_end
        ) ||
        !checked_accumulate_u64(&plan->input_validation_bytes, store->target_bytes) ||
        !checked_accumulate_product_u64(
            &plan->input_validation_bytes,
            UINT64_C(2),
            store->feature_bytes
        )) {
        return PREPARED_RESOURCE_ERROR;
    }
    plan->file_backed_input_bytes = store->file_bytes;
    plan->admitted = 1;
    return PREPARED_SUCCESS;
}

static enum prepared_status authenticate_store(
    const struct session_request *request,
    const struct prepared_store *store
)
{
    unsigned char buffer[64U * 1024U];
    unsigned char store_digest[DIGEST_BYTES];
    unsigned char payload_digest[DIGEST_BYTES];
    struct sha256_context store_context;
    struct sha256_context payload_context;
    uint64_t remaining = store->file_bytes;
    uint64_t position = 0U;
    if (!seek_input((uint64_t)SESSION_HEADER_BYTES)) {
        return PREPARED_INTERNAL_ERROR;
    }
    sha256_init(&store_context);
    sha256_init(&payload_context);
    while (remaining > 0U) {
        const size_t wanted = remaining < (uint64_t)sizeof(buffer)
                                  ? (size_t)remaining
                                  : sizeof(buffer);
        const size_t received = fread(buffer, 1U, wanted, stdin);
        size_t payload_start = 0U;
        if (received != wanted) {
            return ferror(stdin) ? PREPARED_INTERNAL_ERROR : PREPARED_PROTOCOL_ERROR;
        }
        if (!sha256_update(&store_context, buffer, received)) {
            return PREPARED_RESOURCE_ERROR;
        }
        if (position < (uint64_t)STORE_HEADER_BYTES) {
            const uint64_t until_payload = (uint64_t)STORE_HEADER_BYTES - position;
            payload_start = until_payload < (uint64_t)received
                                ? (size_t)until_payload
                                : received;
        }
        if (payload_start < received &&
            !sha256_update(
                &payload_context,
                buffer + payload_start,
                received - payload_start
            )) {
            return PREPARED_RESOURCE_ERROR;
        }
        position += (uint64_t)received;
        remaining -= (uint64_t)received;
    }
    if (fgetc(stdin) != EOF) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if (ferror(stdin)) {
        return PREPARED_INTERNAL_ERROR;
    }
    sha256_final(&store_context, store_digest);
    sha256_final(&payload_context, payload_digest);
    if (memcmp(store_digest, request->store_sha256, DIGEST_BYTES) != 0 ||
        memcmp(payload_digest, store->payload_sha256, DIGEST_BYTES) != 0) {
        return PREPARED_PROTOCOL_ERROR;
    }
    return PREPARED_SUCCESS;
}

static int valid_utf8(const unsigned char *bytes, size_t count)
{
    size_t index = 0U;
    while (index < count) {
        const unsigned char first = bytes[index];
        uint32_t value;
        size_t continuation;
        size_t offset;
        if (first <= 0x7fU) {
            ++index;
            continue;
        }
        if (first >= 0xc2U && first <= 0xdfU) {
            value = (uint32_t)(first & 0x1fU);
            continuation = 1U;
        } else if (first >= 0xe0U && first <= 0xefU) {
            value = (uint32_t)(first & 0x0fU);
            continuation = 2U;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            value = (uint32_t)(first & 0x07U);
            continuation = 3U;
        } else {
            return 0;
        }
        if (continuation > count - index - 1U) {
            return 0;
        }
        for (offset = 1U; offset <= continuation; ++offset) {
            const unsigned char next = bytes[index + offset];
            if ((next & 0xc0U) != 0x80U) {
                return 0;
            }
            value = (value << 6U) | (uint32_t)(next & 0x3fU);
        }
        if ((continuation == 2U && value < UINT32_C(0x800)) ||
            (continuation == 3U && value < UINT32_C(0x10000)) ||
            (value >= UINT32_C(0xd800) && value <= UINT32_C(0xdfff)) ||
            value > UINT32_C(0x10ffff)) {
            return 0;
        }
        index += continuation + 1U;
    }
    return 1;
}

static int lexicographically_before(
    const unsigned char *first,
    size_t first_count,
    const unsigned char *second,
    size_t second_count
)
{
    const size_t common = first_count < second_count ? first_count : second_count;
    const int order = memcmp(first, second, common);
    return order < 0 || (order == 0 && first_count < second_count);
}

static int row_id_has_content(const unsigned char *bytes, size_t count)
{
    size_t index;
    for (index = 0U; index < count; ++index) {
        const unsigned char value = bytes[index];
        if (value != (unsigned char)' ' && value != (unsigned char)'\t' &&
            value != (unsigned char)'\n' && value != (unsigned char)'\v' &&
            value != (unsigned char)'\f' && value != (unsigned char)'\r') {
            return 1;
        }
    }
    return 0;
}

static enum prepared_status validate_row_keys(const struct prepared_store *store)
{
    unsigned char first_id[MAX_ROW_ID_BYTES];
    unsigned char second_id[MAX_ROW_ID_BYTES];
    unsigned char digest[DIGEST_BYTES];
    unsigned char length_bytes[2];
    unsigned char *previous = first_id;
    unsigned char *current = second_id;
    size_t previous_count = 0U;
    uint64_t consumed = 0U;
    size_t row;
    uint64_t absolute;
    if (!checked_add_u64(
            (uint64_t)SESSION_HEADER_BYTES,
            store->row_key_offset,
            &absolute
        ) ||
        !seek_input(absolute)) {
        return PREPARED_INTERNAL_ERROR;
    }
    for (row = 0U; row < (size_t)store->row_count; ++row) {
        uint16_t id_count;
        unsigned char *swap;
        enum prepared_status status;
        if (store->row_key_bytes - consumed < UINT64_C(2)) {
            return PREPARED_PROTOCOL_ERROR;
        }
        status = read_exact(length_bytes, sizeof(length_bytes));
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        consumed += UINT64_C(2);
        id_count = (uint16_t)((uint16_t)length_bytes[0] |
                              ((uint16_t)length_bytes[1] << 8U));
        if (id_count == 0U || (size_t)id_count > MAX_ROW_ID_BYTES ||
            store->row_key_bytes - consumed < (uint64_t)id_count + DIGEST_BYTES) {
            return PREPARED_PROTOCOL_ERROR;
        }
        status = read_exact(current, (size_t)id_count);
        if (status == PREPARED_SUCCESS) {
            status = read_exact(digest, sizeof(digest));
        }
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        consumed += (uint64_t)id_count + DIGEST_BYTES;
        if (!valid_utf8(current, (size_t)id_count) ||
            !row_id_has_content(current, (size_t)id_count) ||
            (row > 0U && !lexicographically_before(
                previous,
                previous_count,
                current,
                (size_t)id_count
            ))) {
            return PREPARED_PROTOCOL_ERROR;
        }
        previous_count = (size_t)id_count;
        swap = previous;
        previous = current;
        current = swap;
    }
    return consumed == store->row_key_bytes
               ? PREPARED_SUCCESS
               : PREPARED_PROTOCOL_ERROR;
}

static void release_buffers(struct prepared_buffers *buffers)
{
    free(buffers->domain_indices);
    free(buffers->targets);
    free(buffers->domain_moments);
    free(buffers->coefficient_means);
    free(buffers->coefficient_scales);
    free(buffers->intercepts);
    free(buffers->weights);
    free(buffers->scores);
    free(buffers->feature_row);
    free(buffers->delta_x);
    free(buffers->delta_y);
    free(buffers->subset_x_mean);
    free(buffers->subset_y_mean);
    free(buffers->subset_xx);
    free(buffers->subset_xy);
    free(buffers->gram);
    free(buffers->cholesky);
    free(buffers->right_hand_sides);
    memset(buffers, 0, sizeof(*buffers));
}

static enum prepared_status allocate_buffers(
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers
)
{
    const size_t domain_cells = plan->domain_count * plan->domain_moment_cells;
    buffers->domain_indices = (unsigned char *)malloc(plan->row_count);
    buffers->targets = (double *)malloc(plan->target_cells * sizeof(double));
    buffers->domain_moments = (double *)calloc(domain_cells, sizeof(double));
    buffers->coefficient_means =
        (double *)malloc(plan->subset_count * CONTINUOUS_WIDTH * sizeof(double));
    buffers->coefficient_scales =
        (double *)malloc(plan->subset_count * CONTINUOUS_WIDTH * sizeof(double));
    buffers->intercepts =
        (double *)malloc(plan->subset_count * plan->target_count * sizeof(double));
    buffers->weights = (double *)malloc(plan->weight_cells * sizeof(double));
    buffers->scores = (double *)malloc(plan->score_cells * sizeof(double));
    buffers->feature_row = (double *)malloc(plan->feature_count * sizeof(double));
    buffers->delta_x = (double *)malloc(plan->feature_count * sizeof(double));
    buffers->delta_y = (double *)malloc(plan->target_count * sizeof(double));
    buffers->subset_x_mean = (double *)malloc(plan->feature_count * sizeof(double));
    buffers->subset_y_mean = (double *)malloc(plan->target_count * sizeof(double));
    buffers->subset_xx = (double *)malloc(plan->packed_count * sizeof(double));
    buffers->subset_xy =
        (double *)malloc(plan->feature_count * plan->target_count * sizeof(double));
    buffers->gram =
        (double *)malloc(plan->maximum_width * plan->maximum_width * sizeof(double));
    buffers->cholesky =
        (double *)malloc(plan->maximum_width * plan->maximum_width * sizeof(double));
    buffers->right_hand_sides =
        (double *)malloc(plan->target_count * plan->maximum_width * sizeof(double));
    if (buffers->domain_indices == NULL || buffers->targets == NULL ||
        buffers->domain_moments == NULL || buffers->coefficient_means == NULL ||
        buffers->coefficient_scales == NULL || buffers->intercepts == NULL ||
        buffers->weights == NULL || buffers->scores == NULL ||
        buffers->feature_row == NULL || buffers->delta_x == NULL ||
        buffers->delta_y == NULL || buffers->subset_x_mean == NULL ||
        buffers->subset_y_mean == NULL || buffers->subset_xx == NULL ||
        buffers->subset_xy == NULL || buffers->gram == NULL ||
        buffers->cholesky == NULL || buffers->right_hand_sides == NULL) {
        return PREPARED_ALLOCATION_ERROR;
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status load_domains_and_targets(
    const struct prepared_store *store,
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers
)
{
    uint64_t observed_counts[MAX_DOMAINS] = {0U, 0U, 0U, 0U, 0U, 0U, 0U};
    unsigned char padding[7];
    uint64_t absolute;
    uint64_t domain_end;
    uint64_t padding_count;
    size_t index;
    enum prepared_status status;
    if (!checked_add_u64(
            (uint64_t)SESSION_HEADER_BYTES,
            store->domain_offset,
            &absolute
        ) ||
        !seek_input(absolute)) {
        return PREPARED_INTERNAL_ERROR;
    }
    status = read_exact(buffers->domain_indices, plan->row_count);
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    for (index = 0U; index < plan->row_count; ++index) {
        const unsigned char domain = buffers->domain_indices[index];
        if ((size_t)domain >= plan->domain_count) {
            return PREPARED_PROTOCOL_ERROR;
        }
        ++observed_counts[domain];
    }
    for (index = 0U; index < plan->domain_count; ++index) {
        if (observed_counts[index] != store->domain_counts[index]) {
            return PREPARED_PROTOCOL_ERROR;
        }
    }
    if (!checked_add_u64(store->domain_offset, store->domain_bytes, &domain_end) ||
        store->feature_offset < domain_end) {
        return PREPARED_PROTOCOL_ERROR;
    }
    padding_count = store->feature_offset - domain_end;
    if (padding_count > UINT64_C(7)) {
        return PREPARED_PROTOCOL_ERROR;
    }
    if (padding_count > 0U) {
        status = read_exact(padding, (size_t)padding_count);
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        if (!all_zero(padding, (size_t)padding_count)) {
            return PREPARED_PROTOCOL_ERROR;
        }
    }
    if (!checked_add_u64(
            (uint64_t)SESSION_HEADER_BYTES,
            store->target_offset,
            &absolute
        ) ||
        !seek_input(absolute)) {
        return PREPARED_INTERNAL_ERROR;
    }
    status = read_exact(
        (unsigned char *)buffers->targets,
        plan->target_cells * sizeof(double)
    );
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    for (index = 0U; index < plan->target_cells; ++index) {
        if (!canonical_f64(buffers->targets[index])) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static size_t packed_index(size_t dimension, size_t row, size_t column)
{
    if (row > column) {
        const size_t swap = row;
        row = column;
        column = swap;
    }
    return row * dimension - row * (row - 1U) / 2U + (column - row);
}

static int finite_long_double_to_double(long double value, double *result)
{
    double converted;
    if (!isfinite(value) || value > (long double)DBL_MAX ||
        value < -(long double)DBL_MAX) {
        return 0;
    }
    converted = (double)value;
    if (!isfinite(converted)) {
        return 0;
    }
    *result = positive_zero(converted);
    return 1;
}

static enum prepared_status update_domain_moments(
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers,
    size_t domain,
    uint64_t previous_count,
    const double *targets
)
{
    double *base = buffers->domain_moments + domain * plan->domain_moment_cells;
    double *x_mean = base;
    double *y_mean = x_mean + plan->feature_count;
    double *xx = y_mean + plan->target_count;
    double *xy = xx + plan->packed_count;
    const uint64_t next_count = previous_count + UINT64_C(1);
    size_t row;
    size_t column;
    size_t target;
    for (row = 0U; row < plan->feature_count; ++row) {
        const long double delta =
            (long double)buffers->feature_row[row] - (long double)x_mean[row];
        if (!finite_long_double_to_double(delta, &buffers->delta_x[row]) ||
            !finite_long_double_to_double(
                (long double)x_mean[row] + delta / (long double)next_count,
                &x_mean[row]
            )) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    for (target = 0U; target < plan->target_count; ++target) {
        const long double delta = (long double)targets[target] - (long double)y_mean[target];
        if (!finite_long_double_to_double(delta, &buffers->delta_y[target]) ||
            !finite_long_double_to_double(
                (long double)y_mean[target] + delta / (long double)next_count,
                &y_mean[target]
            )) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    for (row = 0U; row < plan->feature_count; ++row) {
        for (column = row; column < plan->feature_count; ++column) {
            const size_t position = packed_index(plan->feature_count, row, column);
            const long double updated = (long double)xx[position] +
                (long double)buffers->delta_x[row] *
                    ((long double)buffers->feature_row[column] -
                     (long double)x_mean[column]);
            if (!finite_long_double_to_double(updated, &xx[position])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (target = 0U; target < plan->target_count; ++target) {
            const size_t position = row * plan->target_count + target;
            const long double updated = (long double)xy[position] +
                (long double)buffers->delta_x[row] *
                    ((long double)targets[target] - (long double)y_mean[target]);
            if (!finite_long_double_to_double(updated, &xy[position])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status build_domain_moments(
    const struct prepared_store *store,
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers
)
{
    uint64_t domain_counts[MAX_DOMAINS] = {0U, 0U, 0U, 0U, 0U, 0U, 0U};
    uint64_t domain_masks[MAX_DOMAINS] = {0U, 0U, 0U, 0U, 0U, 0U, 0U};
    uint64_t absolute;
    size_t row;
    if (!checked_add_u64(
            (uint64_t)SESSION_HEADER_BYTES,
            store->feature_offset,
            &absolute
        ) ||
        !seek_input(absolute)) {
        return PREPARED_INTERNAL_ERROR;
    }
    for (row = 0U; row < plan->row_count; ++row) {
        const size_t domain = (size_t)buffers->domain_indices[row];
        const double *target_row = buffers->targets + row * plan->target_count;
        size_t feature;
        enum prepared_status status = read_exact(
            (unsigned char *)buffers->feature_row,
            plan->feature_count * sizeof(double)
        );
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        for (feature = 0U; feature < plan->feature_count; ++feature) {
            const double value = buffers->feature_row[feature];
            if (!canonical_f64(value) ||
                (feature < CONTINUOUS_WIDTH && value < 0.0) ||
                (feature >= CONTINUOUS_WIDTH &&
                 feature < (size_t)UNIVERSAL_SURFACE_WIDTH &&
                 value != 0.0 && value != 1.0)) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (feature = 0U; feature < TAG_COUNT; ++feature) {
            if (buffers->feature_row[TAG_OFFSET + feature] == 1.0) {
                domain_masks[domain] |= UINT64_C(1) << feature;
            }
        }
        status = update_domain_moments(
            plan,
            buffers,
            domain,
            domain_counts[domain],
            target_row
        );
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        ++domain_counts[domain];
    }
    for (row = 0U; row < plan->domain_count; ++row) {
        const double *base =
            buffers->domain_moments + row * plan->domain_moment_cells;
        const double *xx = base + plan->feature_count + plan->target_count;
        size_t feature;
        if (domain_counts[row] != store->domain_counts[row] ||
            domain_masks[row] != store->domain_masks[row]) {
            return PREPARED_PROTOCOL_ERROR;
        }
        for (feature = 0U; feature < plan->feature_count; ++feature) {
            const double diagonal = xx[packed_index(
                plan->feature_count,
                feature,
                feature
            )];
            if (!canonical_f64(diagonal) || diagonal < 0.0) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
    }
    return PREPARED_SUCCESS;
}

static size_t active_raw_index(
    const struct prepared_plan *plan,
    uint64_t tag_mask,
    size_t position
)
{
    size_t tag;
    size_t active_tag_position = TAG_OFFSET;
    (void)plan;
    if (position < TAG_OFFSET) {
        return position;
    }
    for (tag = 0U; tag < TAG_COUNT; ++tag) {
        if ((tag_mask & (UINT64_C(1) << tag)) != 0U) {
            if (active_tag_position == position) {
                return TAG_OFFSET + tag;
            }
            ++active_tag_position;
        }
    }
    return (size_t)UNIVERSAL_SURFACE_WIDTH + (position - active_tag_position);
}

static enum prepared_status combine_subset_moments(
    const struct prepared_plan *plan,
    const struct prepared_buffers *buffers,
    size_t subset_index,
    struct prepared_buffers *workspace
)
{
    const uint64_t subset_mask = plan->subset_masks[subset_index];
    uint64_t combined_count = 0U;
    size_t domain;
    memset(workspace->subset_x_mean, 0, plan->feature_count * sizeof(double));
    memset(workspace->subset_y_mean, 0, plan->target_count * sizeof(double));
    memset(workspace->subset_xx, 0, plan->packed_count * sizeof(double));
    memset(
        workspace->subset_xy,
        0,
        plan->feature_count * plan->target_count * sizeof(double)
    );
    for (domain = 0U; domain < plan->domain_count; ++domain) {
        const double *base;
        const double *right_x_mean;
        const double *right_y_mean;
        const double *right_xx;
        const double *right_xy;
        uint64_t right_count;
        uint64_t next_count;
        long double factor;
        size_t row;
        size_t column;
        size_t target;
        if ((subset_mask & (UINT64_C(1) << domain)) == 0U) {
            continue;
        }
        base = buffers->domain_moments + domain * plan->domain_moment_cells;
        right_x_mean = base;
        right_y_mean = right_x_mean + plan->feature_count;
        right_xx = right_y_mean + plan->target_count;
        right_xy = right_xx + plan->packed_count;
        right_count = plan->domain_rows[domain];
        if (combined_count == 0U) {
            memcpy(
                workspace->subset_x_mean,
                right_x_mean,
                plan->feature_count * sizeof(double)
            );
            memcpy(
                workspace->subset_y_mean,
                right_y_mean,
                plan->target_count * sizeof(double)
            );
            memcpy(workspace->subset_xx, right_xx, plan->packed_count * sizeof(double));
            memcpy(
                workspace->subset_xy,
                right_xy,
                plan->feature_count * plan->target_count * sizeof(double)
            );
            combined_count = right_count;
            continue;
        }
        if (!checked_add_u64(combined_count, right_count, &next_count)) {
            return PREPARED_RESOURCE_ERROR;
        }
        factor = (long double)combined_count * (long double)right_count /
                 (long double)next_count;
        for (row = 0U; row < plan->feature_count; ++row) {
            if (!finite_long_double_to_double(
                    (long double)right_x_mean[row] -
                        (long double)workspace->subset_x_mean[row],
                    &workspace->delta_x[row]
                )) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (target = 0U; target < plan->target_count; ++target) {
            if (!finite_long_double_to_double(
                    (long double)right_y_mean[target] -
                        (long double)workspace->subset_y_mean[target],
                    &workspace->delta_y[target]
                )) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (row = 0U; row < plan->feature_count; ++row) {
            const long double updated =
                (long double)workspace->subset_x_mean[row] +
                (long double)workspace->delta_x[row] * (long double)right_count /
                    (long double)next_count;
            if (!finite_long_double_to_double(updated, &workspace->subset_x_mean[row])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (target = 0U; target < plan->target_count; ++target) {
            const long double updated =
                (long double)workspace->subset_y_mean[target] +
                (long double)workspace->delta_y[target] * (long double)right_count /
                    (long double)next_count;
            if (!finite_long_double_to_double(updated, &workspace->subset_y_mean[target])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (row = 0U; row < plan->feature_count; ++row) {
            for (column = row; column < plan->feature_count; ++column) {
                const size_t position = packed_index(plan->feature_count, row, column);
                const long double updated =
                    (long double)workspace->subset_xx[position] +
                    (long double)right_xx[position] +
                    factor * (long double)workspace->delta_x[row] *
                        (long double)workspace->delta_x[column];
                if (!finite_long_double_to_double(updated, &workspace->subset_xx[position])) {
                    return PREPARED_NUMERIC_ERROR;
                }
            }
            for (target = 0U; target < plan->target_count; ++target) {
                const size_t position = row * plan->target_count + target;
                const long double updated =
                    (long double)workspace->subset_xy[position] +
                    (long double)right_xy[position] +
                    factor * (long double)workspace->delta_x[row] *
                        (long double)workspace->delta_y[target];
                if (!finite_long_double_to_double(updated, &workspace->subset_xy[position])) {
                    return PREPARED_NUMERIC_ERROR;
                }
            }
        }
        combined_count = next_count;
    }
    if (combined_count != plan->subset_rows[subset_index]) {
        return PREPARED_INTERNAL_ERROR;
    }
    for (domain = 0U; domain < plan->feature_count; ++domain) {
        const double diagonal = workspace->subset_xx[packed_index(
            plan->feature_count,
            domain,
            domain
        )];
        if (!canonical_f64(diagonal) || diagonal < 0.0) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status factor_cholesky(double *matrix, double *factor, size_t width)
{
    size_t row;
    size_t column;
    memcpy(factor, matrix, width * width * sizeof(double));
    for (row = 0U; row < width; ++row) {
        for (column = 0U; column <= row; ++column) {
            long double value = (long double)factor[row * width + column];
            size_t inner;
            for (inner = 0U; inner < column; ++inner) {
                value -= (long double)factor[row * width + inner] *
                         (long double)factor[column * width + inner];
                if (!isfinite(value)) {
                    return PREPARED_SOLVE_ERROR;
                }
            }
            if (row == column) {
                if (!(value > 0.0L) ||
                    !finite_long_double_to_double(
                        sqrtl(value),
                        &factor[row * width + column]
                    ) ||
                    !(factor[row * width + column] > 0.0)) {
                    return PREPARED_SOLVE_ERROR;
                }
            } else if (!finite_long_double_to_double(
                           value / (long double)factor[column * width + column],
                           &factor[row * width + column]
                       )) {
                return PREPARED_SOLVE_ERROR;
            }
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status solve_target(
    const double *factor,
    const double *right_hand_side,
    double *solution,
    size_t width
)
{
    size_t row;
    for (row = 0U; row < width; ++row) {
        long double value = (long double)right_hand_side[row];
        size_t inner;
        for (inner = 0U; inner < row; ++inner) {
            value -= (long double)factor[row * width + inner] *
                     (long double)solution[inner];
        }
        if (!finite_long_double_to_double(
                value / (long double)factor[row * width + row],
                &solution[row]
            )) {
            return PREPARED_SOLVE_ERROR;
        }
    }
    for (row = width; row-- > 0U;) {
        long double value = (long double)solution[row];
        size_t inner;
        for (inner = row + 1U; inner < width; ++inner) {
            value -= (long double)factor[inner * width + row] *
                     (long double)solution[inner];
        }
        if (!finite_long_double_to_double(
                value / (long double)factor[row * width + row],
                &solution[row]
            )) {
            return PREPARED_SOLVE_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status verify_solution_residual(
    const double *matrix,
    const double *right_hand_side,
    const double *solution,
    size_t width
)
{
    long double matrix_infinity_norm = 0.0L;
    long double solution_infinity_norm = 0.0L;
    long double rhs_infinity_norm = 0.0L;
    long double residual_infinity_norm = 0.0L;
    long double scale;
    long double tolerance;
    size_t row;
    size_t column;
    for (row = 0U; row < width; ++row) {
        long double row_sum = 0.0L;
        for (column = 0U; column < width; ++column) {
            row_sum += fabsl((long double)matrix[row * width + column]);
            if (!isfinite(row_sum)) {
                return PREPARED_SOLVE_ERROR;
            }
        }
        if (row_sum > matrix_infinity_norm) {
            matrix_infinity_norm = row_sum;
        }
        if (fabsl((long double)solution[row]) > solution_infinity_norm) {
            solution_infinity_norm = fabsl((long double)solution[row]);
        }
        if (fabsl((long double)right_hand_side[row]) > rhs_infinity_norm) {
            rhs_infinity_norm = fabsl((long double)right_hand_side[row]);
        }
    }
    for (row = 0U; row < width; ++row) {
        long double residual = -(long double)right_hand_side[row];
        for (column = 0U; column < width; ++column) {
            residual += (long double)matrix[row * width + column] *
                        (long double)solution[column];
            if (!isfinite(residual)) {
                return PREPARED_SOLVE_ERROR;
            }
        }
        residual = fabsl(residual);
        if (residual > residual_infinity_norm) {
            residual_infinity_norm = residual;
        }
    }
    scale = matrix_infinity_norm * solution_infinity_norm + rhs_infinity_norm;
    if (!isfinite(scale)) {
        return PREPARED_SOLVE_ERROR;
    }
    if (scale < 1.0L) {
        scale = 1.0L;
    }
    tolerance = UINT64_C(4096) * (long double)DBL_EPSILON *
                (long double)(width + 1U) * scale;
    if (!isfinite(tolerance) || residual_infinity_norm > tolerance) {
        return PREPARED_SOLVE_ERROR;
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status solve_one_subset(
    const struct session_request *request,
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers,
    size_t subset_index
)
{
    const size_t width = plan->active_widths[subset_index];
    const uint64_t tags = plan->subset_tag_masks[subset_index];
    double *coefficient_mean =
        buffers->coefficient_means + subset_index * CONTINUOUS_WIDTH;
    double *coefficient_scale =
        buffers->coefficient_scales + subset_index * CONTINUOUS_WIDTH;
    size_t row;
    size_t column;
    size_t target;
    enum prepared_status status = combine_subset_moments(
        plan,
        buffers,
        subset_index,
        buffers
    );
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    for (row = 0U; row < CONTINUOUS_WIDTH; ++row) {
        const double diagonal = buffers->subset_xx[packed_index(
            plan->feature_count,
            row,
            row
        )];
        double scale;
        if (!canonical_f64(diagonal) || diagonal < 0.0) {
            return PREPARED_NUMERIC_ERROR;
        }
        scale = sqrt(diagonal / (double)plan->subset_rows[subset_index]);
        if (!isfinite(scale)) {
            return PREPARED_NUMERIC_ERROR;
        }
        coefficient_mean[row] = positive_zero(buffers->subset_x_mean[row]);
        coefficient_scale[row] = scale > 0.0 ? scale : 1.0;
    }
    for (row = 0U; row < width; ++row) {
        const size_t raw_row = active_raw_index(plan, tags, row);
        const double row_scale = row < CONTINUOUS_WIDTH ? coefficient_scale[row] : 1.0;
        if (raw_row >= plan->feature_count) {
            return PREPARED_INTERNAL_ERROR;
        }
        for (column = 0U; column < width; ++column) {
            const size_t raw_column = active_raw_index(plan, tags, column);
            const double column_scale =
                column < CONTINUOUS_WIDTH ? coefficient_scale[column] : 1.0;
            long double value = (long double)buffers->subset_xx[packed_index(
                plan->feature_count,
                raw_row,
                raw_column
            )] / ((long double)row_scale * (long double)column_scale);
            if (row == column) {
                value += (long double)request->ridge;
            }
            if (!finite_long_double_to_double(value, &buffers->gram[row * width + column])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
    }
    status = factor_cholesky(buffers->gram, buffers->cholesky, width);
    if (status != PREPARED_SUCCESS) {
        return status;
    }
    for (target = 0U; target < plan->target_count; ++target) {
        double *solution = buffers->weights + plan->weight_offsets[subset_index] +
                           target * width;
        long double intercept = (long double)buffers->subset_y_mean[target];
        for (row = 0U; row < width; ++row) {
            const size_t raw_row = active_raw_index(plan, tags, row);
            const double scale = row < CONTINUOUS_WIDTH ? coefficient_scale[row] : 1.0;
            if (!finite_long_double_to_double(
                    (long double)buffers->subset_xy[
                        raw_row * plan->target_count + target
                    ] / (long double)scale,
                    &buffers->right_hand_sides[target * width + row]
                )) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        status = solve_target(
            buffers->cholesky,
            buffers->right_hand_sides + target * width,
            solution,
            width
        );
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        status = verify_solution_residual(
            buffers->gram,
            buffers->right_hand_sides + target * width,
            solution,
            width
        );
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        for (row = 0U; row < width; ++row) {
            const size_t raw_row = active_raw_index(plan, tags, row);
            const double encoded_mean =
                row < CONTINUOUS_WIDTH ? 0.0 : buffers->subset_x_mean[raw_row];
            intercept -= (long double)encoded_mean * (long double)solution[row];
            if (!isfinite(intercept)) {
                return PREPARED_SOLVE_ERROR;
            }
        }
        if (!finite_long_double_to_double(
                intercept,
                &buffers->intercepts[subset_index * plan->target_count + target]
            )) {
            return PREPARED_SOLVE_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status solve_all_subsets(
    const struct session_request *request,
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers
)
{
    size_t subset;
    for (subset = 0U; subset < plan->subset_count; ++subset) {
        const enum prepared_status status =
            solve_one_subset(request, plan, buffers, subset);
        if (status != PREPARED_SUCCESS) {
            return status;
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status score_all_blocks(
    const struct prepared_store *store,
    const struct prepared_plan *plan,
    struct prepared_buffers *buffers
)
{
    uint64_t domain_row[MAX_DOMAINS] = {0U, 0U, 0U, 0U, 0U, 0U, 0U};
    uint64_t absolute;
    size_t row;
    if (!checked_add_u64(
            (uint64_t)SESSION_HEADER_BYTES,
            store->feature_offset,
            &absolute
        ) ||
        !seek_input(absolute)) {
        return PREPARED_INTERNAL_ERROR;
    }
    for (row = 0U; row < plan->row_count; ++row) {
        const size_t domain = (size_t)buffers->domain_indices[row];
        size_t feature;
        size_t subset;
        enum prepared_status status = read_exact(
            (unsigned char *)buffers->feature_row,
            plan->feature_count * sizeof(double)
        );
        if (status != PREPARED_SUCCESS) {
            return status;
        }
        for (feature = 0U; feature < plan->feature_count; ++feature) {
            if (!canonical_f64(buffers->feature_row[feature])) {
                return PREPARED_NUMERIC_ERROR;
            }
        }
        for (subset = 0U; subset < plan->subset_count; ++subset) {
            const int block = plan->block_lookup[subset][domain];
            const size_t width = plan->active_widths[subset];
            const uint64_t tags = plan->subset_tag_masks[subset];
            size_t target;
            size_t destination;
            if (block < 0) {
                continue;
            }
            destination = plan->score_offsets[(size_t)block] +
                          (size_t)domain_row[domain] * plan->target_count;
            for (target = 0U; target < plan->target_count; ++target) {
                const double *target_weights =
                    buffers->weights + plan->weight_offsets[subset] + target * width;
                double score = 0.0;
                size_t position;
                for (position = 0U; position < width; ++position) {
                    const size_t raw = active_raw_index(plan, tags, position);
                    double encoded = buffers->feature_row[raw];
                    double product;
                    if (position < CONTINUOUS_WIDTH) {
                        encoded = (encoded - buffers->coefficient_means[
                            subset * CONTINUOUS_WIDTH + position
                        ]) / buffers->coefficient_scales[
                            subset * CONTINUOUS_WIDTH + position
                        ];
                    }
                    product = encoded * target_weights[position];
                    if (!isfinite(encoded) || !isfinite(product)) {
                        return PREPARED_NUMERIC_ERROR;
                    }
                    score += product;
                    if (!isfinite(score)) {
                        return PREPARED_NUMERIC_ERROR;
                    }
                }
                score += buffers->intercepts[subset * plan->target_count + target];
                if (!isfinite(score)) {
                    return PREPARED_NUMERIC_ERROR;
                }
                buffers->scores[destination + target] = positive_zero(score);
            }
        }
        ++domain_row[domain];
    }
    for (row = 0U; row < plan->domain_count; ++row) {
        if (domain_row[row] != plan->domain_rows[row]) {
            return PREPARED_INTERNAL_ERROR;
        }
    }
    return PREPARED_SUCCESS;
}

static enum prepared_status validate_outputs(
    const struct prepared_plan *plan,
    const struct prepared_buffers *buffers
)
{
    uint64_t cells = 0U;
    size_t index;
    for (index = 0U; index < plan->subset_count * CONTINUOUS_WIDTH; ++index) {
        if (!canonical_f64(buffers->coefficient_means[index]) ||
            !canonical_f64(buffers->coefficient_scales[index]) ||
            !(buffers->coefficient_scales[index] > 0.0)) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    for (index = 0U; index < plan->subset_count * plan->target_count; ++index) {
        if (!canonical_f64(buffers->intercepts[index])) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    for (index = 0U; index < plan->weight_cells; ++index) {
        if (!canonical_f64(buffers->weights[index])) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    for (index = 0U; index < plan->score_cells; ++index) {
        if (!canonical_f64(buffers->scores[index])) {
            return PREPARED_NUMERIC_ERROR;
        }
    }
    if (!checked_multiply_u64(
            (uint64_t)plan->subset_count,
            UINT64_C(6) + (uint64_t)plan->target_count,
            &cells
        ) ||
        !checked_accumulate_u64(&cells, (uint64_t)plan->weight_cells) ||
        !checked_accumulate_u64(&cells, (uint64_t)plan->score_cells) ||
        cells != plan->output_numeric_cells) {
        return PREPARED_INTERNAL_ERROR;
    }
    return PREPARED_SUCCESS;
}

static int write_bytes(const void *payload, size_t count)
{
    return fwrite(payload, 1U, count, stdout) == count;
}

static int write_result_header(
    const struct session_request *request,
    const struct prepared_store *store,
    const struct prepared_plan *plan,
    enum prepared_status status
)
{
    unsigned char header[RESULT_HEADER_BYTES];
    uint64_t ridge_bits = 0U;
    memset(header, 0, sizeof(header));
    memcpy(header, RESULT_MAGIC, sizeof(RESULT_MAGIC));
    encode_u32(header + 8U, UINT32_C(1));
    encode_u32(header + 12U, (uint32_t)status);
    if (request->parsed) {
        memcpy(header + 16U, request->nonce, DIGEST_BYTES);
        memcpy(header + 48U, request->store_sha256, DIGEST_BYTES);
        memcpy(header + 80U, request->binary_sha256, DIGEST_BYTES);
        memcpy(&ridge_bits, &request->ridge, sizeof(ridge_bits));
        encode_u64(header + 192U, ridge_bits);
    }
    if (store->parsed) {
        encode_u64(header + 112U, store->domain_count);
        encode_u64(header + 120U, store->row_count);
        encode_u64(header + 128U, store->feature_count);
        encode_u64(header + 136U, store->target_count);
        memcpy(header + 232U, store->payload_sha256, DIGEST_BYTES);
        memcpy(header + 264U, store->logical_store_sha256, DIGEST_BYTES);
        memcpy(header + 296U, store->source_fit_sha256, DIGEST_BYTES);
        memcpy(header + 328U, store->embedding_snapshot_sha256, DIGEST_BYTES);
        memcpy(header + 360U, store->model_catalogue_sha256, DIGEST_BYTES);
        memcpy(header + 392U, store->graph_sha256, DIGEST_BYTES);
    }
    if (plan->admitted) {
        encode_u64(header + 200U, plan->statistics_work);
        encode_u64(header + 208U, plan->solve_work);
        encode_u64(header + 216U, plan->score_work);
        encode_u64(header + 224U, plan->modeled_heap_bytes);
        encode_u64(header + 424U, plan->input_validation_bytes);
        encode_u64(header + 432U, plan->output_numeric_cells);
        encode_u64(header + 440U, plan->file_backed_input_bytes);
    }
    if (status == PREPARED_SUCCESS) {
        encode_u64(header + 144U, (uint64_t)plan->subset_count);
        encode_u64(header + 152U, (uint64_t)plan->block_count);
        encode_u64(header + 160U, plan->score_memberships);
        encode_u64(header + 168U, plan->coefficient_section_bytes);
        encode_u64(header + 176U, plan->score_section_bytes);
        encode_u64(header + 184U, plan->result_bytes);
    } else {
        encode_u64(header + 184U, (uint64_t)RESULT_HEADER_BYTES);
    }
    return write_bytes(header, sizeof(header));
}

static int write_coefficient_records(
    const struct prepared_plan *plan,
    const struct prepared_buffers *buffers
)
{
    size_t subset;
    for (subset = 0U; subset < plan->subset_count; ++subset) {
        unsigned char prefix[48];
        const size_t width = plan->active_widths[subset];
        uint64_t payload_cells = UINT64_C(6) + (uint64_t)plan->target_count;
        uint64_t payload_bytes;
        memset(prefix, 0, sizeof(prefix));
        if (!checked_accumulate_product_u64(
                &payload_cells,
                (uint64_t)plan->target_count,
                (uint64_t)width
            ) ||
            !checked_multiply_u64(payload_cells, UINT64_C(8), &payload_bytes)) {
            return 0;
        }
        encode_u32(prefix, (uint32_t)subset);
        encode_u64(prefix + 8U, plan->subset_masks[subset]);
        encode_u64(prefix + 16U, plan->subset_rows[subset]);
        encode_u64(prefix + 24U, plan->subset_tag_masks[subset]);
        encode_u64(prefix + 32U, (uint64_t)width);
        encode_u64(prefix + 40U, payload_bytes);
        if (!write_bytes(prefix, sizeof(prefix)) ||
            !write_bytes(
                buffers->coefficient_means + subset * CONTINUOUS_WIDTH,
                CONTINUOUS_WIDTH * sizeof(double)
            ) ||
            !write_bytes(
                buffers->coefficient_scales + subset * CONTINUOUS_WIDTH,
                CONTINUOUS_WIDTH * sizeof(double)
            ) ||
            !write_bytes(
                buffers->intercepts + subset * plan->target_count,
                plan->target_count * sizeof(double)
            ) ||
            !write_bytes(
                buffers->weights + plan->weight_offsets[subset],
                plan->target_count * width * sizeof(double)
            )) {
            return 0;
        }
    }
    return 1;
}

static int write_score_records(
    const struct prepared_plan *plan,
    const struct prepared_buffers *buffers
)
{
    size_t block;
    for (block = 0U; block < plan->block_count; ++block) {
        unsigned char prefix[32];
        uint64_t cells;
        uint64_t payload_bytes;
        memset(prefix, 0, sizeof(prefix));
        if (!checked_multiply_u64(
                plan->block_rows[block],
                (uint64_t)plan->target_count,
                &cells
            ) ||
            !checked_multiply_u64(cells, UINT64_C(8), &payload_bytes)) {
            return 0;
        }
        encode_u32(prefix, (uint32_t)block);
        encode_u32(prefix + 4U, (uint32_t)plan->block_subset[block]);
        encode_u32(prefix + 8U, (uint32_t)plan->block_domain[block]);
        encode_u64(prefix + 16U, plan->block_rows[block]);
        encode_u64(prefix + 24U, payload_bytes);
        if (!write_bytes(prefix, sizeof(prefix)) ||
            !write_bytes(
                buffers->scores + plan->score_offsets[block],
                (size_t)cells * sizeof(double)
            )) {
            return 0;
        }
    }
    return 1;
}

static int write_success_result(
    const struct session_request *request,
    const struct prepared_store *store,
    const struct prepared_plan *plan,
    const struct prepared_buffers *buffers
)
{
    return write_result_header(request, store, plan, PREPARED_SUCCESS) &&
           write_coefficient_records(plan, buffers) &&
           write_score_records(plan, buffers) && fflush(stdout) == 0;
}

int main(void)
{
    struct session_request request;
    struct prepared_store store;
    struct prepared_plan plan;
    struct prepared_buffers buffers;
    enum prepared_status status;
    int output_ok;
    memset(&request, 0, sizeof(request));
    memset(&store, 0, sizeof(store));
    memset(&plan, 0, sizeof(plan));
    memset(&buffers, 0, sizeof(buffers));
    if (!platform_is_supported() || !configure_binary_streams()) {
        status = PREPARED_INTERNAL_ERROR;
    } else {
        status = read_session_header(&request);
    }
    if (status == PREPARED_SUCCESS) {
        status = read_store_header(&store);
    }
    if (status == PREPARED_SUCCESS) {
        status = preflight_session(&request, &store, &plan);
    }
    if (status == PREPARED_SUCCESS) {
        status = authenticate_store(&request, &store);
    }
    if (status == PREPARED_SUCCESS) {
        status = validate_row_keys(&store);
    }
    if (status == PREPARED_SUCCESS) {
        status = allocate_buffers(&plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        status = load_domains_and_targets(&store, &plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        status = build_domain_moments(&store, &plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        status = solve_all_subsets(&request, &plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        status = score_all_blocks(&store, &plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        status = validate_outputs(&plan, &buffers);
    }
    if (status == PREPARED_SUCCESS) {
        output_ok = write_success_result(&request, &store, &plan, &buffers);
    } else {
        output_ok = write_result_header(&request, &store, &plan, status) &&
                    fflush(stdout) == 0;
    }
    release_buffers(&buffers);
    return status == PREPARED_SUCCESS && output_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
