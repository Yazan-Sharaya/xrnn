// Layers that accept double data type. For comments explaining parts of the code please refer to layers_f.c.
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif


inline size_t fourDIndexTo1D(size_t n, size_t h, size_t w, size_t c, size_t nf, size_t hf, size_t wf, size_t cf) {
	return hf * wf * cf * n + wf * cf * h + cf * w + c;
}


EXPORT void convForwardD(
	const double* inputs,
	const double* kernels,
	const double* biases,
	double* output,
	size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, size_t ic, const bool nhwc) {
	int b;
	#pragma omp parallel for
	for (b = 0; b < bs; b++) {
		for (size_t k = 0; k < nk; k++) {
			for (size_t h = 0; h < nh; h++) {
				for (size_t w = 0; w < nw; w++) {

					const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
					double tot = 0.;
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


EXPORT void convBackwardD(
	const double* inputs,
	const double* kernels,
	const double* d_values,
	double* d_weights,
	double* d_inputs,
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
					const double d_filters = d_values[index];
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
					const double d_filters = d_values[index];
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


EXPORT void maxPoolForwardD(const double* inputs, double* masks, double* output,
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
					double max = inputs[start_index];
					size_t max_indices[2] = { start_h, start_w };
					for (size_t lh = start_h; lh < end_h; lh++) {
						for (size_t lw = start_w; lw < end_w; lw++) {
							size_t input_index;
							if (nhwc)
								input_index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
							else
								input_index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
							double curr_value = inputs[input_index];
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
					masks[masks_index] = 1.;
				}
			}
		}
	}
}


EXPORT void maxPoolBackwardD(const double* d_values, const double* masks, double* d_inputs,
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
					const double d_filter = d_values[filter_index];
					bool should_break = false;
					for (size_t lh = start_h; lh < end_h; lh++) {
						for (size_t lw = start_w; lw < end_w; lw++) {
							size_t index;
							if (nhwc)
								index = fourDIndexTo1D(b, lh, lw, k, bs, ih, iw, nk);
							else
								index = fourDIndexTo1D(b, k, lh, lw, bs, nk, ih, iw);
							if (masks[index] == 1.) {
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


EXPORT void avgPoolForwardD(const double* inputs, double* output,
	size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
	#pragma omp parallel for
	for (b = 0; b < bs; b++) {
		for (size_t h = 0; h < nh; h++) {
			for (size_t w = 0; w < nw; w++) {
				for (size_t k = 0; k < nk; k++) {

					const size_t start_h = h * sh, start_w = w * sw, end_h = h * sh + kh, end_w = w * sw + kw;
					double sum = 0., n_elements = ((end_h - start_h) * (end_w - start_w));
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


EXPORT void avgPoolBackwardD(const double* d_values, double* d_inputs,
	size_t kh, size_t kw, size_t sh, size_t sw, size_t bs, size_t nh, size_t nw, size_t nk, size_t ih, size_t iw, const bool nhwc) {
    int b;
	double dx_dm = 1. / (kh * kw);
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
					const double d_filter = d_values[filter_index] * dx_dm;
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
