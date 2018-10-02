//
// Created by Joel Jonsson on 27.09.18.
//

#ifndef LIBAPR_DATAPARALLELOPS_HPP
#define LIBAPR_DATAPARALLELOPS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>

//#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

//#include "PyPixelData.hpp"
#include "PyAPRFiltering.hpp"
#include "PyAPR.hpp"

#include <typeinfo>

namespace py = pybind11;

class DataParallelOps {

public:

    void convolve3x3(py::array &apr_list, py::array &input_features, py::array &weights, py::array &bias, py::array &output, py::array &level_delta) {

        PyAPRFiltering filter_fns;

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto dlvl_ptr = (int32 *) dlvl_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=3 || width != 3) {
            std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size;
        size_t in, out, bn;

#pragma omp parallel for schedule(dynamic) private(in, out, bn)
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR<float> *aPyAPR = apr_ptr[bn].cast<PyAPR<float> *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.apr_tree.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /**** initialize and fill the apr tree ****/

                ExtraParticleData<float> tree_data;

                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, aPyAPR->apr.apr_tree, input_ptr, tree_data, in_offset, current_max_level);

                for (out = 0; out < out_channels; ++out) {

                    //std::cout << "hello from thread " << omp_get_thread_num() << std::endl;
                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    float b = bias_ptr[out];

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        size_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;
                        int idx = 0;

                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                //stencil_vec[n].at(y, x, 0) = weights_ptr[offset + idx];
                                stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];

                                idx++;
                            }
                        }
                    }
                    filter_fns.convolve3x3_loop_unrolled_alt(aPyAPR->apr, input_ptr, stencil_vec, b, out, in, bn,
                                                             current_max_level, in_offset, tree_data, apr_iterator,
                                                             tree_iterator, number_in_channels, out_offset, output_ptr);
                }
            }
        }
    }


    void convolve3x3_2D(py::array &apr_list, py::array &input_features, py::array &weights, py::array &bias, py::array &output, py::array &level_delta) {

        PyAPRFiltering filter_fns;

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto dlvl_ptr = (int32 *) dlvl_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=3 || width != 3) {
            std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size;
        size_t in, out, bn;

#pragma omp parallel for schedule(dynamic) private(in, out, bn)
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR<float> *aPyAPR = apr_ptr[bn].cast<PyAPR<float> *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.apr_tree.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /**** initialize and fill the apr tree ****/

                ExtraParticleData<float> tree_data;

                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, aPyAPR->apr.apr_tree, input_ptr, tree_data, in_offset, current_max_level);

                for (out = 0; out < out_channels; ++out) {

                    //std::cout << "hello from thread " << omp_get_thread_num() << std::endl;
                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    float b = bias_ptr[out];

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        size_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;
                        int idx = 0;

                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                //stencil_vec[n].at(y, x, 0) = weights_ptr[offset + idx];
                                stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];

                                idx++;
                            }
                        }
                    }
                    filter_fns.convolve3x3_loop_unrolled_alt(aPyAPR->apr, input_ptr, stencil_vec, b, out, in, bn,
                                                             current_max_level, in_offset, tree_data, apr_iterator,
                                                             tree_iterator, number_in_channels, out_offset, output_ptr);
                }
            }
        }
    }


    /*
    void convolve3x3_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array &weights, py::array &grad_input, py::array &grad_weights, py::array &grad_bias, py::array &level_delta) {

        PyAPRFiltering filter_fns;

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
        auto grad_output_ptr = (float *) grad_output_buf.ptr;
        auto dlvl_ptr = (int32 *) dlvl_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        /// allocate a large array to avoid race condition on the weight gradients
        size_t num_threads = omp_get_max_threads();
        std::vector<float> temp_vec_dw;
        temp_vec_dw.resize(num_threads*out_channels*in_channels*nstencils*9, 0);

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=3 || width != 3) {
            std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size();
        size_t in, out, bn;

#pragma omp parallel for schedule(dynamic) private(in, out, bn)
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR<float> *aPyAPR = apr_ptr[bn].cast<PyAPR<float> *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.apr_tree.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /// initialize and fill the apr tree

                ExtraParticleData<float> tree_data;

                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, aPyAPR->apr.apr_tree, input_ptr, tree_data, in_offset, current_max_level);

                for (out = 0; out < out_channels; ++out) {

                    //std::cout << "hello from thread " << omp_get_thread_num() << std::endl;
                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    const uint64_t dw_offset = omp_get_thread_num() * out_channels * in_channels *  nstencils * 9 +
                                               out * in_channels * nstencils * 9 +
                                               in * nstencils * 9;

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        size_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;
                        int idx = 0;

                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                //stencil_vec[n].at(y, x, 0) = weights_ptr[offset + idx];
                                stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];

                                idx++;
                            }
                        }
                    }
                    filter_fns.convolve3x3_loop_unrolled_alt(aPyAPR->apr, input_ptr, stencil_vec, b, out, in, bn,
                                                             current_max_level, in_offset, tree_data, apr_iterator,
                                                             tree_iterator, number_in_channels, out_offset, output_ptr);
                }
            }

        }
    }
    */
};



#endif //LIBAPR_DATAPARALLELOPS_HPP
