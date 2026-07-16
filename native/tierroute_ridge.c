/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Project-owned, dependency-free C11 centered-ridge sidecar.
 *
 * Standard input and output are the versioned binary protocol documented in
 * docs/native-ridge-protocol.md.  In particular, stdout never carries text.
 * The implementation is deliberately sequential so one authenticated binary
 * produces deterministic results for a fixed compiler/platform build.
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

#define REQUEST_HEADER_BYTES ((size_t)80)
#define RESPONSE_HEADER_BYTES ((size_t)64)
#define REQUEST_ID_BYTES ((size_t)32)
#define ACCUMULATION_BLOCK ((size_t)16)

#define MAX_SAMPLE_COUNT UINT64_C(1000000)
#define MAX_FEATURE_COUNT UINT64_C(4096)
#define MAX_TARGET_COUNT UINT64_C(256)
#define MAX_REQUEST_BYTES UINT64_C(4294967296)
#define MAX_RESPONSE_BYTES UINT64_C(16777216)
#define MAX_ALLOCATION_BYTES UINT64_C(2147483648)
#define MAX_WORK_UNITS UINT64_C(32000000000)

enum ridge_status {
    RIDGE_SUCCESS = 0,
    RIDGE_PROTOCOL_ERROR = 1,
    RIDGE_RESOURCE_ERROR = 2,
    RIDGE_NUMERIC_ERROR = 3,
    RIDGE_ALLOCATION_ERROR = 4,
    RIDGE_SOLVE_ERROR = 5,
    RIDGE_INTERNAL_ERROR = 6
};

struct ridge_request {
    unsigned char request_id[REQUEST_ID_BYTES];
    uint64_t sample_count;
    uint64_t feature_count;
    uint64_t target_count;
    double ridge;
    int header_parsed;
};

struct ridge_shape {
    size_t sample_count;
    size_t feature_count;
    size_t target_count;
    size_t feature_cells;
    size_t target_cells;
    size_t square_cells;
    size_t coefficient_cells;
};

struct ridge_buffers {
    double *features;
    double *targets;
    double *gram;
    double *cholesky;
    double *right_hand_sides;
    double *weights;
    double *feature_means;
    double *feature_scales;
    double *target_means;
    double *target_scales;
    double *intercepts;
};

static const unsigned char REQUEST_MAGIC[8] = {'T', 'R', 'R', 'I', 'D', 'G', '0', '1'};
static const unsigned char RESPONSE_MAGIC[8] = {'T', 'R', 'R', 'R', 'E', 'S', '0', '1'};

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

static int checked_accumulate_product(
    uint64_t *total,
    uint64_t first,
    uint64_t second
)
{
    uint64_t product;
    uint64_t updated;

    if (!checked_multiply_u64(first, second, &product) ||
        !checked_add_u64(*total, product, &updated)) {
        return 0;
    }
    *total = updated;
    return 1;
}

static uint32_t decode_u32_little_endian(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0]) |
           ((uint32_t)bytes[1] << 8U) |
           ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

static uint64_t decode_u64_little_endian(const unsigned char *bytes)
{
    uint64_t value = 0U;
    size_t index;

    for (index = 0U; index < 8U; ++index) {
        value |= ((uint64_t)bytes[index]) << (8U * index);
    }
    return value;
}

static void encode_u32_little_endian(unsigned char *bytes, uint32_t value)
{
    size_t index;

    for (index = 0U; index < 4U; ++index) {
        bytes[index] = (unsigned char)((value >> (8U * index)) & UINT32_C(255));
    }
}

static void encode_u64_little_endian(unsigned char *bytes, uint64_t value)
{
    size_t index;

    for (index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)((value >> (8U * index)) & UINT64_C(255));
    }
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

static enum ridge_status read_exact_bytes(unsigned char *destination, size_t byte_count)
{
    size_t offset = 0U;

    while (offset < byte_count) {
        const size_t received = fread(destination + offset, 1U, byte_count - offset, stdin);
        if (received > 0U) {
            offset += received;
            continue;
        }
        if (ferror(stdin)) {
            return RIDGE_INTERNAL_ERROR;
        }
        return RIDGE_PROTOCOL_ERROR;
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status read_request_header(struct ridge_request *request)
{
    unsigned char header[REQUEST_HEADER_BYTES];
    uint64_t ridge_bits;
    enum ridge_status status;

    memset(header, 0, sizeof(header));
    status = read_exact_bytes(header, sizeof(header));
    if (status != RIDGE_SUCCESS) {
        return status;
    }

    memcpy(request->request_id, header + 16U, REQUEST_ID_BYTES);
    request->sample_count = decode_u64_little_endian(header + 48U);
    request->feature_count = decode_u64_little_endian(header + 56U);
    request->target_count = decode_u64_little_endian(header + 64U);
    ridge_bits = decode_u64_little_endian(header + 72U);
    memcpy(&request->ridge, &ridge_bits, sizeof(request->ridge));
    request->header_parsed = 1;

    if (memcmp(header, REQUEST_MAGIC, sizeof(REQUEST_MAGIC)) != 0 ||
        decode_u32_little_endian(header + 8U) != UINT32_C(1) ||
        decode_u32_little_endian(header + 12U) != UINT32_C(0)) {
        return RIDGE_PROTOCOL_ERROR;
    }
    return RIDGE_SUCCESS;
}

static int add_work_term(uint64_t *work, uint64_t term)
{
    uint64_t updated;

    if (!checked_add_u64(*work, term, &updated)) {
        return 0;
    }
    *work = updated;
    return 1;
}

static enum ridge_status preflight_request(
    const struct ridge_request *request,
    struct ridge_shape *shape
)
{
    const uint64_t n = request->sample_count;
    const uint64_t d = request->feature_count;
    const uint64_t m = request->target_count;
    uint64_t feature_cells;
    uint64_t target_cells;
    uint64_t square_cells;
    uint64_t coefficient_cells;
    uint64_t dimensions_sum;
    uint64_t triangular_twice;
    uint64_t triangular;
    uint64_t request_cells;
    uint64_t request_bytes;
    uint64_t response_cells;
    uint64_t response_bytes;
    uint64_t allocation_cells = 0U;
    uint64_t allocation_bytes;
    uint64_t work = 0U;
    uint64_t term;

    if (n == 0U || n > MAX_SAMPLE_COUNT || d == 0U || d > MAX_FEATURE_COUNT ||
        m == 0U || m > MAX_TARGET_COUNT) {
        return RIDGE_RESOURCE_ERROR;
    }
    if (!checked_multiply_u64(n, d, &feature_cells) ||
        !checked_multiply_u64(n, m, &target_cells) ||
        !checked_multiply_u64(d, d, &square_cells) ||
        !checked_multiply_u64(m, d, &coefficient_cells) ||
        !checked_add_u64(d, m, &dimensions_sum) ||
        !checked_add_u64(d, UINT64_C(1), &term) ||
        !checked_multiply_u64(d, term, &triangular_twice)) {
        return RIDGE_RESOURCE_ERROR;
    }
    triangular = triangular_twice / UINT64_C(2);

    if (!checked_add_u64(feature_cells, target_cells, &request_cells) ||
        !checked_multiply_u64(request_cells, UINT64_C(8), &request_bytes) ||
        !checked_add_u64(request_bytes, (uint64_t)REQUEST_HEADER_BYTES, &request_bytes) ||
        request_bytes > MAX_REQUEST_BYTES) {
        return RIDGE_RESOURCE_ERROR;
    }
    if (!checked_add_u64(coefficient_cells, m, &response_cells) ||
        !checked_multiply_u64(response_cells, UINT64_C(8), &response_bytes) ||
        !checked_add_u64(response_bytes, (uint64_t)RESPONSE_HEADER_BYTES, &response_bytes) ||
        response_bytes > MAX_RESPONSE_BYTES) {
        return RIDGE_RESOURCE_ERROR;
    }

    /* X + Y + Gram + Cholesky + RHS + weights + means/scales/intercepts. */
    if (!add_work_term(&allocation_cells, feature_cells) ||
        !add_work_term(&allocation_cells, target_cells) ||
        !checked_accumulate_product(&allocation_cells, UINT64_C(2), square_cells) ||
        !checked_accumulate_product(&allocation_cells, UINT64_C(2), coefficient_cells) ||
        !checked_accumulate_product(&allocation_cells, UINT64_C(2), d) ||
        !checked_accumulate_product(&allocation_cells, UINT64_C(3), m) ||
        !checked_multiply_u64(allocation_cells, UINT64_C(8), &allocation_bytes) ||
        allocation_bytes > MAX_ALLOCATION_BYTES) {
        return RIDGE_RESOURCE_ERROR;
    }

    /*
     * Reviewed deterministic work estimate:
     * 3*n*(d+m) + n*d*(d+1)/2 + n*d*m + d^3 + 2*m*d^2 + m*d.
     * It conservatively charges the factorization as d^3, then separately
     * charges both triangular solves and residual verification.
     */
    if (!checked_multiply_u64(n, dimensions_sum, &term) ||
        !checked_multiply_u64(term, UINT64_C(3), &term) || !add_work_term(&work, term) ||
        !checked_multiply_u64(n, triangular, &term) || !add_work_term(&work, term) ||
        !checked_multiply_u64(feature_cells, m, &term) || !add_work_term(&work, term) ||
        !checked_multiply_u64(square_cells, d, &term) || !add_work_term(&work, term) ||
        !checked_multiply_u64(square_cells, m, &term) ||
        !checked_multiply_u64(term, UINT64_C(2), &term) || !add_work_term(&work, term) ||
        !add_work_term(&work, coefficient_cells) || work > MAX_WORK_UNITS) {
        return RIDGE_RESOURCE_ERROR;
    }

    if (feature_cells > (uint64_t)(SIZE_MAX / sizeof(double)) ||
        target_cells > (uint64_t)(SIZE_MAX / sizeof(double)) ||
        square_cells > (uint64_t)(SIZE_MAX / sizeof(double)) ||
        coefficient_cells > (uint64_t)(SIZE_MAX / sizeof(double)) ||
        d > (uint64_t)(SIZE_MAX / sizeof(double)) ||
        m > (uint64_t)(SIZE_MAX / sizeof(double))) {
        return RIDGE_RESOURCE_ERROR;
    }

    shape->sample_count = (size_t)n;
    shape->feature_count = (size_t)d;
    shape->target_count = (size_t)m;
    shape->feature_cells = (size_t)feature_cells;
    shape->target_cells = (size_t)target_cells;
    shape->square_cells = (size_t)square_cells;
    shape->coefficient_cells = (size_t)coefficient_cells;
    return RIDGE_SUCCESS;
}

static void release_buffers(struct ridge_buffers *buffers)
{
    free(buffers->features);
    free(buffers->targets);
    free(buffers->gram);
    free(buffers->cholesky);
    free(buffers->right_hand_sides);
    free(buffers->weights);
    free(buffers->feature_means);
    free(buffers->feature_scales);
    free(buffers->target_means);
    free(buffers->target_scales);
    free(buffers->intercepts);
    memset(buffers, 0, sizeof(*buffers));
}

static enum ridge_status allocate_buffers(
    const struct ridge_shape *shape,
    struct ridge_buffers *buffers
)
{
    buffers->features = (double *)malloc(shape->feature_cells * sizeof(double));
    buffers->targets = (double *)malloc(shape->target_cells * sizeof(double));
    buffers->gram = (double *)malloc(shape->square_cells * sizeof(double));
    buffers->cholesky = (double *)malloc(shape->square_cells * sizeof(double));
    buffers->right_hand_sides =
        (double *)malloc(shape->coefficient_cells * sizeof(double));
    buffers->weights = (double *)malloc(shape->coefficient_cells * sizeof(double));
    buffers->feature_means = (double *)malloc(shape->feature_count * sizeof(double));
    buffers->feature_scales = (double *)malloc(shape->feature_count * sizeof(double));
    buffers->target_means = (double *)malloc(shape->target_count * sizeof(double));
    buffers->target_scales = (double *)malloc(shape->target_count * sizeof(double));
    buffers->intercepts = (double *)malloc(shape->target_count * sizeof(double));

    if (buffers->features == NULL || buffers->targets == NULL || buffers->gram == NULL ||
        buffers->cholesky == NULL || buffers->right_hand_sides == NULL ||
        buffers->weights == NULL || buffers->feature_means == NULL ||
        buffers->feature_scales == NULL || buffers->target_means == NULL ||
        buffers->target_scales == NULL || buffers->intercepts == NULL) {
        return RIDGE_ALLOCATION_ERROR;
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status read_double_cells(double *destination, size_t count)
{
    size_t offset = 0U;

    while (offset < count) {
        const size_t received = fread(
            destination + offset,
            sizeof(double),
            count - offset,
            stdin
        );
        if (received > 0U) {
            offset += received;
            continue;
        }
        if (ferror(stdin)) {
            return RIDGE_INTERNAL_ERROR;
        }
        return RIDGE_PROTOCOL_ERROR;
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status read_payload(
    const struct ridge_shape *shape,
    struct ridge_buffers *buffers
)
{
    enum ridge_status status;
    int trailing;
    size_t index;

    status = read_double_cells(buffers->features, shape->feature_cells);
    if (status == RIDGE_SUCCESS) {
        status = read_double_cells(buffers->targets, shape->target_cells);
    }
    if (status != RIDGE_SUCCESS) {
        return status;
    }

    trailing = fgetc(stdin);
    if (trailing != EOF) {
        return RIDGE_PROTOCOL_ERROR;
    }
    if (ferror(stdin)) {
        return RIDGE_INTERNAL_ERROR;
    }
    for (index = 0U; index < shape->feature_cells; ++index) {
        if (!isfinite(buffers->features[index])) {
            return RIDGE_NUMERIC_ERROR;
        }
    }
    for (index = 0U; index < shape->target_cells; ++index) {
        if (!isfinite(buffers->targets[index])) {
            return RIDGE_NUMERIC_ERROR;
        }
    }
    return RIDGE_SUCCESS;
}

static int convert_finite_long_double(long double value, double *converted)
{
    double result;

    if (!isfinite(value) || value > (long double)DBL_MAX || value < -(long double)DBL_MAX) {
        return 0;
    }
    result = (double)value;
    if (!isfinite(result)) {
        return 0;
    }
    *converted = result;
    return 1;
}

static enum ridge_status center_columns_welford(
    double *matrix,
    size_t row_count,
    size_t column_count,
    double *means,
    double *scales
)
{
    size_t column;
    size_t row;

    /*
     * Shift each column by its overflow-safe min/max midpoint, then scale the
     * remaining deviations into [-1, 1] before Welford's recurrence.  The
     * shift preserves low-order differences in a large-offset column, while
     * the scale prevents a finite-but-wide column from overflowing.  Neither
     * operation changes the ridge problem; both are used only for the mean.
     */
    for (column = 0U; column < column_count; ++column) {
        double minimum = matrix[column];
        double maximum = matrix[column];
        double midpoint;
        double scale = 0.0;
        long double normalized_mean = 0.0L;
        long double unscaled_mean;
        long double minimum_deviation;
        long double maximum_deviation;

        for (row = 1U; row < row_count; ++row) {
            const double value = matrix[row * column_count + column];
            if (value < minimum) {
                minimum = value;
            }
            if (value > maximum) {
                maximum = value;
            }
        }
        midpoint = minimum * 0.5 + maximum * 0.5;
        if (!isfinite(midpoint)) {
            return RIDGE_NUMERIC_ERROR;
        }
        minimum_deviation = (long double)minimum - (long double)midpoint;
        maximum_deviation = (long double)maximum - (long double)midpoint;
        if (!isfinite(minimum_deviation) || !isfinite(maximum_deviation) ||
            fabsl(minimum_deviation) > (long double)DBL_MAX ||
            fabsl(maximum_deviation) > (long double)DBL_MAX) {
            return RIDGE_NUMERIC_ERROR;
        }
        scale = (double)fmaxl(fabsl(minimum_deviation), fabsl(maximum_deviation));
        scales[column] = scale;
        if (scale != 0.0) {
            for (row = 0U; row < row_count; ++row) {
                const long double value =
                    ((long double)matrix[row * column_count + column] -
                     (long double)midpoint) /
                    (long double)scale;
                const long double delta = value - normalized_mean;
                normalized_mean += delta / (long double)(row + 1U);
                if (!isfinite(normalized_mean)) {
                    return RIDGE_NUMERIC_ERROR;
                }
            }
        }
        unscaled_mean =
            (long double)midpoint + normalized_mean * (long double)scale;
        if (!convert_finite_long_double(unscaled_mean, &means[column])) {
            return RIDGE_NUMERIC_ERROR;
        }
        for (row = 0U; row < row_count; ++row) {
            const long double centered =
                (long double)matrix[row * column_count + column] -
                (long double)means[column];
            if (!convert_finite_long_double(
                    centered,
                    &matrix[row * column_count + column]
                )) {
                return RIDGE_NUMERIC_ERROR;
            }
        }
    }
    return RIDGE_SUCCESS;
}

static int stable_add_long_double(
    long double value,
    long double *sum,
    long double *correction
)
{
    const long double updated = *sum + value;
    long double correction_term;

    if (!isfinite(value) || !isfinite(updated)) {
        return 0;
    }
    if (fabsl(*sum) >= fabsl(value)) {
        correction_term = (*sum - updated) + value;
    } else {
        correction_term = (value - updated) + *sum;
    }
    *correction += correction_term;
    if (!isfinite(*correction)) {
        return 0;
    }
    *sum = updated;
    return 1;
}

static enum ridge_status build_gram_matrix(
    const struct ridge_shape *shape,
    const double *features,
    double *gram
)
{
    long double sums[ACCUMULATION_BLOCK * ACCUMULATION_BLOCK];
    long double corrections[ACCUMULATION_BLOCK * ACCUMULATION_BLOCK];
    size_t feature_block;
    size_t paired_block;

    for (feature_block = 0U; feature_block < shape->feature_count;
         feature_block += ACCUMULATION_BLOCK) {
        const size_t feature_width =
            (shape->feature_count - feature_block < ACCUMULATION_BLOCK)
                ? shape->feature_count - feature_block
                : ACCUMULATION_BLOCK;
        for (paired_block = 0U; paired_block <= feature_block;
             paired_block += ACCUMULATION_BLOCK) {
            const size_t paired_width =
                (shape->feature_count - paired_block < ACCUMULATION_BLOCK)
                    ? shape->feature_count - paired_block
                    : ACCUMULATION_BLOCK;
            const int diagonal_block = feature_block == paired_block;
            size_t local_index;
            size_t row;
            size_t local_feature;

            for (local_index = 0U; local_index < ACCUMULATION_BLOCK * ACCUMULATION_BLOCK;
                 ++local_index) {
                sums[local_index] = 0.0L;
                corrections[local_index] = 0.0L;
            }
            for (row = 0U; row < shape->sample_count; ++row) {
                const double *row_values = features + row * shape->feature_count;
                for (local_feature = 0U; local_feature < feature_width; ++local_feature) {
                    const size_t paired_limit = diagonal_block
                                                    ? local_feature + 1U
                                                    : paired_width;
                    size_t local_paired;
                    for (local_paired = 0U; local_paired < paired_limit; ++local_paired) {
                        const size_t accumulator =
                            local_feature * ACCUMULATION_BLOCK + local_paired;
                        const long double product =
                            (long double)row_values[feature_block + local_feature] *
                            (long double)row_values[paired_block + local_paired];
                        if (!stable_add_long_double(
                                product,
                                &sums[accumulator],
                                &corrections[accumulator]
                            )) {
                            return RIDGE_NUMERIC_ERROR;
                        }
                    }
                }
            }
            for (local_feature = 0U; local_feature < feature_width; ++local_feature) {
                const size_t paired_limit = diagonal_block
                                                ? local_feature + 1U
                                                : paired_width;
                size_t local_paired;
                for (local_paired = 0U; local_paired < paired_limit; ++local_paired) {
                    const size_t accumulator =
                        local_feature * ACCUMULATION_BLOCK + local_paired;
                    const size_t feature = feature_block + local_feature;
                    const size_t paired = paired_block + local_paired;
                    double converted;
                    if (!convert_finite_long_double(
                            sums[accumulator] + corrections[accumulator],
                            &converted
                        )) {
                        return RIDGE_NUMERIC_ERROR;
                    }
                    gram[feature * shape->feature_count + paired] = converted;
                    gram[paired * shape->feature_count + feature] = converted;
                }
            }
        }
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status build_right_hand_sides(
    const struct ridge_shape *shape,
    const double *features,
    const double *targets,
    double *right_hand_sides
)
{
    long double sums[ACCUMULATION_BLOCK * ACCUMULATION_BLOCK];
    long double corrections[ACCUMULATION_BLOCK * ACCUMULATION_BLOCK];
    size_t feature_block;
    size_t target_block;

    for (feature_block = 0U; feature_block < shape->feature_count;
         feature_block += ACCUMULATION_BLOCK) {
        const size_t feature_width =
            (shape->feature_count - feature_block < ACCUMULATION_BLOCK)
                ? shape->feature_count - feature_block
                : ACCUMULATION_BLOCK;
        for (target_block = 0U; target_block < shape->target_count;
             target_block += ACCUMULATION_BLOCK) {
            const size_t target_width =
                (shape->target_count - target_block < ACCUMULATION_BLOCK)
                    ? shape->target_count - target_block
                    : ACCUMULATION_BLOCK;
            size_t local_index;
            size_t row;
            size_t local_feature;

            for (local_index = 0U; local_index < ACCUMULATION_BLOCK * ACCUMULATION_BLOCK;
                 ++local_index) {
                sums[local_index] = 0.0L;
                corrections[local_index] = 0.0L;
            }
            for (row = 0U; row < shape->sample_count; ++row) {
                const double *feature_row = features + row * shape->feature_count;
                const double *target_row = targets + row * shape->target_count;
                for (local_feature = 0U; local_feature < feature_width; ++local_feature) {
                    size_t local_target;
                    for (local_target = 0U; local_target < target_width; ++local_target) {
                        const size_t accumulator =
                            local_feature * ACCUMULATION_BLOCK + local_target;
                        const long double product =
                            (long double)feature_row[feature_block + local_feature] *
                            (long double)target_row[target_block + local_target];
                        if (!stable_add_long_double(
                                product,
                                &sums[accumulator],
                                &corrections[accumulator]
                            )) {
                            return RIDGE_NUMERIC_ERROR;
                        }
                    }
                }
            }
            for (local_feature = 0U; local_feature < feature_width; ++local_feature) {
                size_t local_target;
                for (local_target = 0U; local_target < target_width; ++local_target) {
                    const size_t accumulator =
                        local_feature * ACCUMULATION_BLOCK + local_target;
                    const size_t target = target_block + local_target;
                    const size_t feature = feature_block + local_feature;
                    if (!convert_finite_long_double(
                            sums[accumulator] + corrections[accumulator],
                            &right_hand_sides[target * shape->feature_count + feature]
                        )) {
                        return RIDGE_NUMERIC_ERROR;
                    }
                }
            }
        }
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status add_ridge_and_factor(
    const struct ridge_request *request,
    const struct ridge_shape *shape,
    double *gram,
    double *cholesky
)
{
    size_t row;
    size_t column;

    for (row = 0U; row < shape->feature_count; ++row) {
        double diagonal;
        if (!convert_finite_long_double(
                (long double)gram[row * shape->feature_count + row] +
                    (long double)request->ridge,
                &diagonal
            ) ||
            diagonal <= 0.0) {
            return RIDGE_SOLVE_ERROR;
        }
        gram[row * shape->feature_count + row] = diagonal;
    }
    memcpy(cholesky, gram, shape->square_cells * sizeof(double));

    for (row = 0U; row < shape->feature_count; ++row) {
        for (column = 0U; column <= row; ++column) {
            long double value = (long double)cholesky[row * shape->feature_count + column];
            size_t inner;
            for (inner = 0U; inner < column; ++inner) {
                value -=
                    (long double)cholesky[row * shape->feature_count + inner] *
                    (long double)cholesky[column * shape->feature_count + inner];
                if (!isfinite(value)) {
                    return RIDGE_SOLVE_ERROR;
                }
            }
            if (row == column) {
                long double root;
                if (!(value > 0.0L)) {
                    return RIDGE_SOLVE_ERROR;
                }
                root = sqrtl(value);
                if (!convert_finite_long_double(
                        root,
                        &cholesky[row * shape->feature_count + column]
                    ) ||
                    cholesky[row * shape->feature_count + column] <= 0.0) {
                    return RIDGE_SOLVE_ERROR;
                }
            } else {
                const long double quotient =
                    value /
                    (long double)cholesky[column * shape->feature_count + column];
                if (!convert_finite_long_double(
                        quotient,
                        &cholesky[row * shape->feature_count + column]
                    )) {
                    return RIDGE_SOLVE_ERROR;
                }
            }
        }
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status solve_all_targets(
    const struct ridge_shape *shape,
    const double *cholesky,
    const double *right_hand_sides,
    double *weights
)
{
    size_t target;

    for (target = 0U; target < shape->target_count; ++target) {
        size_t row;
        for (row = 0U; row < shape->feature_count; ++row) {
            long double value =
                (long double)right_hand_sides[target * shape->feature_count + row];
            size_t inner;
            for (inner = 0U; inner < row; ++inner) {
                value -=
                    (long double)cholesky[row * shape->feature_count + inner] *
                    (long double)weights[target * shape->feature_count + inner];
                if (!isfinite(value)) {
                    return RIDGE_SOLVE_ERROR;
                }
            }
            value /= (long double)cholesky[row * shape->feature_count + row];
            if (!convert_finite_long_double(
                    value,
                    &weights[target * shape->feature_count + row]
                )) {
                return RIDGE_SOLVE_ERROR;
            }
        }
        for (row = shape->feature_count; row-- > 0U;) {
            long double value =
                (long double)weights[target * shape->feature_count + row];
            size_t inner;
            for (inner = row + 1U; inner < shape->feature_count; ++inner) {
                value -=
                    (long double)cholesky[inner * shape->feature_count + row] *
                    (long double)weights[target * shape->feature_count + inner];
                if (!isfinite(value)) {
                    return RIDGE_SOLVE_ERROR;
                }
            }
            value /= (long double)cholesky[row * shape->feature_count + row];
            if (!convert_finite_long_double(
                    value,
                    &weights[target * shape->feature_count + row]
                )) {
                return RIDGE_SOLVE_ERROR;
            }
        }
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status verify_residuals(
    const struct ridge_shape *shape,
    const double *gram,
    const double *right_hand_sides,
    const double *weights
)
{
    long double matrix_infinity_norm = 0.0L;
    size_t row;
    size_t target;

    for (row = 0U; row < shape->feature_count; ++row) {
        long double row_sum = 0.0L;
        size_t column;
        for (column = 0U; column < shape->feature_count; ++column) {
            row_sum += fabsl((long double)gram[row * shape->feature_count + column]);
            if (!isfinite(row_sum)) {
                return RIDGE_SOLVE_ERROR;
            }
        }
        if (row_sum > matrix_infinity_norm) {
            matrix_infinity_norm = row_sum;
        }
    }

    for (target = 0U; target < shape->target_count; ++target) {
        long double weight_infinity_norm = 0.0L;
        long double rhs_infinity_norm = 0.0L;
        long double residual_infinity_norm = 0.0L;
        long double scale;
        long double tolerance;
        size_t column;

        for (column = 0U; column < shape->feature_count; ++column) {
            const long double weight =
                fabsl((long double)weights[target * shape->feature_count + column]);
            const long double rhs =
                fabsl((long double)right_hand_sides[
                    target * shape->feature_count + column
                ]);
            if (weight > weight_infinity_norm) {
                weight_infinity_norm = weight;
            }
            if (rhs > rhs_infinity_norm) {
                rhs_infinity_norm = rhs;
            }
        }
        for (row = 0U; row < shape->feature_count; ++row) {
            long double residual =
                -(long double)right_hand_sides[target * shape->feature_count + row];
            for (column = 0U; column < shape->feature_count; ++column) {
                residual +=
                    (long double)gram[row * shape->feature_count + column] *
                    (long double)weights[target * shape->feature_count + column];
                if (!isfinite(residual)) {
                    return RIDGE_SOLVE_ERROR;
                }
            }
            residual = fabsl(residual);
            if (residual > residual_infinity_norm) {
                residual_infinity_norm = residual;
            }
        }
        scale = matrix_infinity_norm * weight_infinity_norm + rhs_infinity_norm;
        if (!isfinite(scale)) {
            return RIDGE_SOLVE_ERROR;
        }
        if (scale < 1.0L) {
            scale = 1.0L;
        }
        /* Scale-aware backward-error allowance for dense Cholesky in binary64. */
        tolerance = UINT64_C(4096) * (long double)DBL_EPSILON *
                    (long double)(shape->feature_count + 1U) * scale;
        if (!isfinite(tolerance) || residual_infinity_norm > tolerance) {
            return RIDGE_SOLVE_ERROR;
        }
    }
    return RIDGE_SUCCESS;
}

static enum ridge_status recover_intercepts(
    const struct ridge_shape *shape,
    const double *feature_means,
    const double *target_means,
    const double *weights,
    double *intercepts
)
{
    size_t target;

    for (target = 0U; target < shape->target_count; ++target) {
        long double dot_sum = 0.0L;
        long double correction = 0.0L;
        size_t feature;
        for (feature = 0U; feature < shape->feature_count; ++feature) {
            const long double product =
                (long double)feature_means[feature] *
                (long double)weights[target * shape->feature_count + feature];
            if (!stable_add_long_double(product, &dot_sum, &correction)) {
                return RIDGE_SOLVE_ERROR;
            }
        }
        if (!convert_finite_long_double(
                (long double)target_means[target] - (dot_sum + correction),
                &intercepts[target]
            )) {
            return RIDGE_SOLVE_ERROR;
        }
    }
    return RIDGE_SUCCESS;
}

static int write_response_header(
    const struct ridge_request *request,
    enum ridge_status status
)
{
    unsigned char header[RESPONSE_HEADER_BYTES];

    memset(header, 0, sizeof(header));
    memcpy(header, RESPONSE_MAGIC, sizeof(RESPONSE_MAGIC));
    encode_u32_little_endian(header + 8U, UINT32_C(1));
    encode_u32_little_endian(header + 12U, (uint32_t)status);
    if (request->header_parsed) {
        memcpy(header + 16U, request->request_id, REQUEST_ID_BYTES);
        encode_u64_little_endian(header + 48U, request->feature_count);
        encode_u64_little_endian(header + 56U, request->target_count);
    }
    return fwrite(header, 1U, sizeof(header), stdout) == sizeof(header);
}

static int write_response(
    const struct ridge_request *request,
    const struct ridge_shape *shape,
    const struct ridge_buffers *buffers,
    enum ridge_status status
)
{
    if (!write_response_header(request, status)) {
        return 0;
    }
    if (status == RIDGE_SUCCESS) {
        if (fwrite(
                buffers->intercepts,
                sizeof(double),
                shape->target_count,
                stdout
            ) != shape->target_count ||
            fwrite(
                buffers->weights,
                sizeof(double),
                shape->coefficient_cells,
                stdout
            ) != shape->coefficient_cells) {
            return 0;
        }
    }
    return fflush(stdout) == 0;
}

static void report_status(enum ridge_status status)
{
    static const char *const messages[] = {
        "",
        "tierroute_ridge: protocol error\n",
        "tierroute_ridge: resource bound refused\n",
        "tierroute_ridge: invalid numeric input\n",
        "tierroute_ridge: allocation failed\n",
        "tierroute_ridge: solve or residual failed\n",
        "tierroute_ridge: I/O, platform, or internal error\n"
    };

    if (status >= RIDGE_PROTOCOL_ERROR && status <= RIDGE_INTERNAL_ERROR) {
        (void)fputs(messages[(size_t)status], stderr);
    }
}

int main(void)
{
    struct ridge_request request;
    struct ridge_shape shape;
    struct ridge_buffers buffers;
    enum ridge_status status = RIDGE_SUCCESS;

    memset(&request, 0, sizeof(request));
    memset(&shape, 0, sizeof(shape));
    memset(&buffers, 0, sizeof(buffers));

    if (!configure_binary_streams() || !platform_is_supported()) {
        status = RIDGE_INTERNAL_ERROR;
    }
    if (status == RIDGE_SUCCESS) {
        status = read_request_header(&request);
    }
    if (status == RIDGE_SUCCESS) {
        if (!isfinite(request.ridge) || !(request.ridge > 0.0)) {
            status = RIDGE_NUMERIC_ERROR;
        }
    }
    if (status == RIDGE_SUCCESS) {
        status = preflight_request(&request, &shape);
    }
    if (status == RIDGE_SUCCESS) {
        status = allocate_buffers(&shape, &buffers);
    }
    if (status == RIDGE_SUCCESS) {
        status = read_payload(&shape, &buffers);
    }
    if (status == RIDGE_SUCCESS) {
        status = center_columns_welford(
            buffers.features,
            shape.sample_count,
            shape.feature_count,
            buffers.feature_means,
            buffers.feature_scales
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = center_columns_welford(
            buffers.targets,
            shape.sample_count,
            shape.target_count,
            buffers.target_means,
            buffers.target_scales
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = build_gram_matrix(&shape, buffers.features, buffers.gram);
    }
    if (status == RIDGE_SUCCESS) {
        status = build_right_hand_sides(
            &shape,
            buffers.features,
            buffers.targets,
            buffers.right_hand_sides
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = add_ridge_and_factor(
            &request,
            &shape,
            buffers.gram,
            buffers.cholesky
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = solve_all_targets(
            &shape,
            buffers.cholesky,
            buffers.right_hand_sides,
            buffers.weights
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = verify_residuals(
            &shape,
            buffers.gram,
            buffers.right_hand_sides,
            buffers.weights
        );
    }
    if (status == RIDGE_SUCCESS) {
        status = recover_intercepts(
            &shape,
            buffers.feature_means,
            buffers.target_means,
            buffers.weights,
            buffers.intercepts
        );
    }

    if (!write_response(&request, &shape, &buffers, status)) {
        status = RIDGE_INTERNAL_ERROR;
    }
    release_buffers(&buffers);
    if (status != RIDGE_SUCCESS) {
        report_status(status);
        return (int)status;
    }
    return 0;
}
