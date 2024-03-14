// Layers that accept float data type.
#include <omp.h>
#include <stdbool.h>  // defines bool in C.
#include <stddef.h>  // defines size_t

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)  // This is only required on Windows and is used to make a function callable from outside the DLL.
#else
    #define EXPORT
#endif


inline size_t fourDIndexTo1D(size_t n, size_t h, size_t w, size_t c, size_t nf, size_t hf, size_t wf, size_t cf) {
    return hf * wf * cf * n + wf * cf * h + cf * w + c;
}


EXPORT void convForwardF(
    const float* inputs,
    const float* kernels,
    const float* biases,
    float* output,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, size_t ic, const bool nhwc) {
    int b;  // Need to initialize the iteration variable `b` before the omp parallel section because MSVC support for C is really old,
    // as disused here https://www.reddit.com/r/C_Programming/comments/13v32z1/pragma_omp_parallel_for_not_compiling_when_it/
    // This weirdly only applies to C and C++.
    #pragma omp parallel for  // Using int data type instead of size_t (which is an unsinged integer) because microsoft openmp doesn't support uint as index variables.
    for (b = 0; b < bs; b++) {  // select single image from batch. Used int instead of size_t because omp doesn't work with unsigned integer indices on MSVC 2022 for some reason.
        for (size_t k = 0; k < nk; k++) {  // loop over the kernels which are going to be the new channels of the input.
            for (size_t h = 0; h < nh; h++) {
                for (size_t w = 0; w < nw; w++) {

                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    float tot = 0.f;
                    size_t lkh = 0;
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        size_t lkw = 0;
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            for (size_t c = 0; c < ic; c++) {
                                size_t input_index;
                                if (nhwc)  // NHWC format.
                                    input_index = fourDIndexTo1D(b, lh, lw, c, bs, ih, iw, ic);
                                else  // NCHW format.
                                    input_index = fourDIndexTo1D(b, c, lh, lw, bs, ic, ih, iw);
                                const size_t kernel_index = fourDIndexTo1D(k, lkh, lkw, c, nk, kh, kw, ic);
                                tot += inputs[input_index] * kernels[kernel_index];
                            }
                            lkw++;
                        }
                        lkh++;
                    }
                    tot += biases[k];
                    if (nhwc)
                        output[fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk)] = tot;
                    else
                        output[fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw)] = tot;
                }
            }
        }
    }
}


EXPORT void convBackwardF(
    const float* inputs,
    const float* kernels,
    const float* d_values,
    float* d_weights,
    float* d_inputs,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, size_t ic, const bool nhwc) {
    int b;
    #pragma omp parallel for
    for (b = 0; b < bs; b++) {
        for (size_t k = 0; k < nk; k++) {
            for (size_t h = 0; h < nh; h++) {
                for (size_t w = 0; w < nw; w++) {

                    size_t index;
                    if (nhwc)
                        index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                    else
                        index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                    const float d_filters = d_values[index];
                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    size_t lkh = 0;
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        size_t lkw = 0;
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            for (size_t c = 0; c < ic; c++) {
                                size_t input_index;
                                if (nhwc)
                                    input_index = fourDIndexTo1D(b, lh, lw, c, bs, ih, iw, ic);
                                else
                                    input_index = fourDIndexTo1D(b, c, lh, lw, bs, ic, ih, iw);
                                const size_t kernel_index = fourDIndexTo1D(k, lkh, lkw, c, nk, kh, kw, ic);
                                d_inputs[input_index] += kernels[kernel_index] * d_filters;
                            }
                            lkw++;
                        }
                        lkh++;
                    }
                }
            }
        }
    }

    // Okay so this looks dumb and inefficient, but it's actually as fast as updating both d_weights with d_inputs in the
    // above loop on a two core cpu, and this version becomes faster as the number of cores increases.
    // The reason I did this is that I'm currently compiling with msvc and using its outdated openmp support (version 2.0!)
    // and that version of openmp doesn't support array reduction, so I had to separate the reduction operation in two loops
    // to be able to parallelize the operations on both of the arrays.
    // Having them in the same loop with parallelization returns wrong results and using shared directives makes the code run
    // slower than if it was compiled without parallelization.
    int k;
    #pragma omp parallel for
    for (k = 0; k < nk; k++) {
        for (size_t b = 0; b < bs; b++) {
            for (size_t h = 0; h < nh; h++) {
                for (size_t w = 0; w < nw; w++) {

                    size_t index;
                    if (nhwc)
                        index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                    else
                        index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                    const float d_filters = d_values[index];
                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    size_t lkh = 0;
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        size_t lkw = 0;
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            for (size_t c = 0; c < ic; c++) {
                                size_t input_index;
                                if (nhwc)
                                    input_index = fourDIndexTo1D(b, lh, lw, c, bs, ih, iw, ic);
                                else
                                    input_index = fourDIndexTo1D(b, c, lh, lw, bs, ic, ih, iw);
                                const size_t kernel_index = fourDIndexTo1D(k, lkh, lkw, c, nk, kh, kw, ic);
                                d_weights[kernel_index] += inputs[input_index] * d_filters;
                            }
                            lkw++;
                        }
                        lkh++;
                    }
                }
            }
        }
    }
}


EXPORT void maxPoolForwardF(const float* inputs, float* masks, float* output,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
    #pragma omp parallel for
    for (b = 0; b < bs; b++) {
        for (size_t h = 0; h < nh; h++) {
            for (size_t w = 0; w < nw; w++) {
                for (size_t k = 0; k < nk; k++) {

                    size_t output_index, masks_index, start_index;
                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    if (nhwc)
                        start_index = fourDIndexTo1D(b, start_h, start_w, k, bs, ih, iw, nk);
                    else
                        start_index = fourDIndexTo1D(b, k, start_h, start_w, bs, nk, ih, iw);
                    float max = inputs[start_index];
                    size_t max_indices[2] = { start_h, start_w };
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            size_t input_index;
                            if (nhwc)
                                input_index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
                            else
                                input_index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
                            float curr_value = inputs[input_index];
                            if (curr_value > max) {
                                max = curr_value;
                                max_indices[0] = lh;
                                max_indices[1] = lw;
                            }
                        }
                    }
                    if (nhwc) {
                        output_index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                        masks_index = fourDIndexTo1D(b, max_indices[0], max_indices[1], k, bs, ih, iw, nk);
                    }
                    else {
                        output_index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                        masks_index = fourDIndexTo1D(b, k, max_indices[0], max_indices[1], bs, nk, ih, iw);
                    }
                    output[output_index] = max;
                    masks[masks_index] = 1.f;
                }
            }
        }
    }
}


EXPORT void maxPoolBackwardF(const float* d_values, const float* masks, float* d_inputs,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
    #pragma omp parallel for
    for (b = 0; b < bs; b++) {
        for (size_t h = 0; h < nh; h++) {
            for (size_t w = 0; w < nw; w++) {
                for (size_t k = 0; k < nk; k++) {

                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    size_t filter_index;
                    if (nhwc)
                        filter_index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                    else
                        filter_index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                    const float d_filter = d_values[filter_index];
                    bool should_break = false;
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            size_t index;
                            if (nhwc)
                                index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
                            else
                                index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
                            if (masks[index] == 1.f) {
                                d_inputs[index] = d_filter;
                                should_break = true;  // no need to continue looping because there's only one max value per window.
                                break;
                            }
                        }
                    if (should_break)
                        break;
                    }
                }
            }
        }
    }
}


EXPORT void avgPoolForwardF(const float* inputs, float* output,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
    #pragma omp parallel for
    for (b = 0; b < bs; b++) {
        for (size_t h = 0; h < nh; h++) {
            for (size_t w = 0; w < nw; w++) {
                for (size_t k = 0; k < nk; k++) {

                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    float sum = 0.f, n_elements = ((end_h - start_h) * (end_w - start_w));
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            size_t input_index;
                            if (nhwc)
                                input_index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
                            else
                                input_index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
                            sum += inputs[input_index];
                        }
                    }
                    size_t output_index;
                    if (nhwc)
                        output_index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                    else
                        output_index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                    output[output_index] = sum / n_elements;
                }
            }
        }
    }
}


EXPORT void avgPoolBackwardF(const float* d_values, float* d_inputs,
    size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
    float dx_dm = 1.f / (kh * kw);
    #pragma omp parallel for
    for (b = 0; b < bs; b++) {
        for (size_t h = 0; h < nh; h++) {
            for (size_t w = 0; w < nw; w++) {
                for (size_t k = 0; k < nk; k++) {

                    const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
                    size_t filter_index;
                    if (nhwc)
                        filter_index = fourDIndexTo1D(b, h, w, k, bs, nh, nw, nk);
                    else
                        filter_index = fourDIndexTo1D(b, k, h, w, bs, nk, nh, nw);
                    const float d_filter = d_values[filter_index] * dx_dm;
                    for (size_t lh = start_h; lh < end_h; lh++) {
                        for (size_t lw = start_w; lw < end_w; lw++) {
                            size_t output_index;
                            if (nhwc)
                                output_index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
                            else
                                output_index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
                            d_inputs[output_index] += d_filter;
                        }
                    }
                }
            }
        }
    }
}
