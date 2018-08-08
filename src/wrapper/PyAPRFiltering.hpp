//
// Created by Joel Jonsson on 26.07.18.
//

#ifndef LIBAPR_PYAPRFILTERING_HPP
#define LIBAPR_PYAPRFILTERING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"

#include "PyAPR.hpp"
#include "PyPixelData.hpp"


namespace py = pybind11;


class PyAPRFiltering {

public:

    template<typename ImageType, typename T>
    py::array_t<float> convolve_equivalent(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, float bias) {

        //output_intensities.resize(input_intensities.size());

        py::buffer_info input_buf = input_intensities.request();

        auto output = py::array_t<float_t>(input_buf.size);
        py::buffer_info output_buf = output.request(true);

        //auto inptr = (float *) input_buf.ptr;
        auto outptr = (float *) output_buf.ptr;

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, tree_data/*this argument is not used*/, tree_data);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        int stencil_counter = 0;

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries


            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_iterator,
                                   tree_iterator,
                                   tree_data,
                                   temp_vec,
                                   input_intensities,
                                   stencil_shape,
                                   stencil_half);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                       temp_vec, input_intensities, stencil_shape, stencil_half);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                                  temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    counter++;

                                }
                            }
                        }

                        outptr[apr_iterator.global_index()] = neigh_sum + bias;//std::roundf(neigh_sum/(norm*1.0f));

                    }//y, pixels/columns
                }//x , rows
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        return output;
    }



    template<typename ImageType, typename T>
    void convolve_equivalent_loop(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, float bias, py::array &output, int out_channel, int in_channel) {

        py::buffer_info input_buf = input_intensities.request();

        int number_in_channels = input_buf.shape[0];
        int nparticles = input_buf.shape[1];

        uint64_t in_offset = in_channel * nparticles;

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info output_buf = output.request(true);
        auto output_ptr = (float *) output_buf.ptr;

        uint64_t out_offset = out_channel * apr.total_number_particles();
        int stencil_counter = 0;

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries


            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array2(level,
                                    padd,
                                    apr,
                                    apr_iterator,
                                    tree_iterator,
                                    tree_data,
                                    temp_vec,
                                    input_intensities,
                                    stencil_shape,
                                    stencil_half,
                                    in_offset);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array2(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                                  temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    counter++;

                                }
                            }
                        }

                        if(in_channel == number_in_channels-1) {
                            neigh_sum += bias;
                        }

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum;

                    }//y, pixels/columns
                }//x , rows
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels
        //return output;
    }


    template<typename ImageType>
    void convolve_ds_stencil_loop(APR<ImageType> &apr,
                                  py::array &particle_intensities,
                                  PixelData<float> &inputStencil,
                                  float bias,
                                  py::array &output,
                                  int out_channel,
                                  int in_channel) {

        std::vector<PixelData<float>> stencil_vec;
        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;
        stencil_vec.resize(nstencils);

        /// first stencil (pixel resolution) is used as given
        stencil_vec[0].swap(inputStencil);

        /// stencils at lower levels (resolution) are downsampled versions of the input stencil
        for(int level_delta = 1; level_delta<nstencils; ++level_delta) {
            downsample_stencil_alt(stencil_vec[0], stencil_vec[level_delta], level_delta, false, true);
        }

        /*
        for( int i = 0; i<nstencils; ++i){
            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil_dlevel" + std::to_string(i) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil_vec[i]);
        }
        */


        convolve_equivalent_loop(apr, particle_intensities, stencil_vec, bias, output, out_channel, in_channel);

    }


    template<typename ImageType>
    py::array_t<float> convolve_ds_stencil(APR<ImageType> &apr,
                                           py::array &particle_intensities,
                                           py::array &inputStencil,
                                           py::array &bias,
                                           bool normalize=true) {

        std::vector<PixelData<float>> stencil_vec;

        py::buffer_info stencil_buf = inputStencil.request();
        //py::buffer_info intensity_buf = particle_intensities.request();

        auto stenc_ptr = (float *)stencil_buf.ptr;

        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;
        stencil_vec.resize(nstencils);

        PixelData<float> stencilCopy;

        switch ((int)stencil_buf.ndim) {

            case 1:
                stencilCopy.init(stencil_buf.shape[0], 1, 1);
                break;
            case 2:
                stencilCopy.init(stencil_buf.shape[0], stencil_buf.shape[1], 1);
                break;
            case 3:
                stencilCopy.init(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2]);
                break;
            default:
                std::cerr <<"could not correctly read input convolution kernel in call to PyAPRFiltering::convolve_ds_stencil" << std::endl;
        }

        for(ssize_t i=0; i<stencil_buf.size; ++i) {
            stencilCopy.mesh[i] = stenc_ptr[i];
        }

        stencil_vec[0].swap(stencilCopy); // first stencil is a copy of the input stencil

        // remaining stencils are downsampled versions of the input stencil
        for(int level_delta = 1; level_delta<nstencils; ++level_delta) {
            downsample_stencil_alt(stencil_vec[0], stencil_vec[level_delta], level_delta, normalize, true);
        }

        for( int i = 0; i<nstencils; ++i){
            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil_dlevel" + std::to_string(i) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil_vec[i]);
        }

        float b = ((float *)bias.request().ptr)[0];

        return convolve_equivalent(apr, particle_intensities, stencil_vec, b);

    }

/*
    template<typename ImageType>
    py::array_t<float> convolve_ds_cnn(APR<ImageType> &apr,
                                       py::array &input_features,
                                       py::array &input_weights,
                                       py::array &input_bias,
                                       bool normalize=true) {

        std::vector<PixelData<float>> stencil_vec;

        py::buffer_info weights_buf = input_weights.request();
        //py::buffer_info intensity_buf = particle_intensities.request();

        auto weights_ptr = (float *)weights_buf.ptr;

        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;
        int out_channels = weights_buf.shape[0];

        stencil_vec.resize(nstencils * out_channels); // holds out_channels * nlevels convolution kernels

        int kernelsize = weights_buf.shape[1] * weights_buf.shape[2] * weights_buf.shape[3];
        int offset = 0;

        for(int i = 0; i < out_channels; ++ i) {
            stencil_vec[i].init(weights_buf.shape[2], weights_buf.shape[3], weights_buf.shape[1]);

            for(int idx = 0; idx < kernelsize; ++idx) {
                stencil_vec[i].mesh[idx] = weights_ptr[idx + offset];
            }

            offset += kernelsize;
        }

        // the given convolution kernels are used at pixel resolution, while downsampled versions of them are used for
        // lower resolution particles
        for( int out = 0; out < out_channels; ++out) {
            for (int level_delta = 1; level_delta < nstencils; ++level_delta) {
                int offset = level_delta * out_channels;
                downsample_stencil_cnn2d(stencil_vec[out], stencil_vec[offset + out], level_delta, normalize, true);
            }
        }

        for( int i = 0; i<nstencils; ++i){
            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil_dlevel" + std::to_string(i) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil_vec[i]);
        }

        return aprconv2d(apr, particle_intensities, stencil_vec, b, out_channels);

    }
    */

/*
    template<typename ImageType, typename T>
    py::array_t<float> aprconv2d(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, py::array &bias, ssize_t out_channels) {

        //output_intensities.resize(input_intensities.size());

        py::buffer_info input_buf = input_intensities.request();

        std::vector<ssize_t> outshape = {input_buf.shape[0], out_channels};

        auto output = py::array_t<float_t>(outshape);
        py::buffer_info output_buf = output.request(true);

        //auto inptr = (float *) input_buf.ptr;
        auto outptr = (float *) output_buf.ptr;

        /// initialize and fill the apr tree
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, tree_data, tree_data);

        /// iterators for accessing apr data
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        int stencil_counter = 0;

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries


            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_iterator,
                                   tree_iterator,
                                   tree_data,
                                   temp_vec,
                                   input_intensities,
                                   stencil_shape,
                                   stencil_half);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                       temp_vec, input_intensities, stencil_shape, stencil_half);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                                  temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    counter++;
                                }
                            }
                        }

                        outptr[apr_iterator.global_index()] = neigh_sum + bias;//std::roundf(neigh_sum/(norm*1.0f));

                    }//y, pixels/columns
                }//x , rows
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        return output;
    }
    */

    template<typename ImageType, typename S, typename T>
    void convolve(APR<ImageType> &apr, const PixelData<T>& stencil, ExtraParticleData<S> &particle_intensities, ExtraParticleData<float> &conv_particle_intensities, uint16_t level_delta) {

        conv_particle_intensities.init(particle_intensities.total_number_particles());

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, particle_intensities, tree_data);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        const std::vector<int> stencil_shape = {(int) stencil.y_num,
                                                (int) stencil.x_num,
                                                (int) stencil.z_num};
        const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                               (stencil_shape[2] - 1) / 2};

        // assert stencil_shape compatible with apr org_dims?

        int level = std::max((int)apr_iterator.level_min(), (int)apr_iterator.level_max() - (int)level_delta);
        unsigned int z = 0;
        unsigned int x = 0;

        const int z_num = apr_iterator.spatial_index_z_max(level);

        const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                               stencil_shape[0] - 1 : 1;
        const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                               stencil_shape[1] - 1 : 1;

        PixelData<float> temp_vec;
        temp_vec.init(y_num_m,
                      x_num_m,
                      stencil_shape[2],
                      0); //zero padded boundaries

        //initial condition
        for (int padd = 0; padd < stencil_half[2]; ++padd) {
            update_dense_array(level,
                               padd,
                               apr,
                               apr_iterator,
                               tree_iterator,
                               tree_data,
                               temp_vec,
                               particle_intensities,
                               stencil_shape,
                               stencil_half);
        }

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

            if (z < (z_num - stencil_half[2])) {
                //update the next z plane for the access
                update_dense_array(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                   temp_vec, particle_intensities, stencil_shape, stencil_half);
            } else {
                //padding
                uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                for (x = 0; x < temp_vec.x_num; ++x) {
                    std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                              temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                }
            }

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    float neigh_sum = 0;
                    int counter = 0;

                    const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                    const int i = x + stencil_half[1];

                    //compute the stencil

                    for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                        for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                            for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                neigh_sum += (stencil.mesh[counter] *
                                              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                counter++;
                            }
                        }
                    }

                    conv_particle_intensities[apr_iterator] = neigh_sum;

                }//y, pixels/columns
            }//x , rows
        }//z
    }


    template<typename ImageType>
    void update_dense_array(const uint64_t level,
                            const uint64_t z,
                            APR<ImageType> &apr,
                            APRIterator &apr_iterator,
                            APRTreeIterator &treeIterator,
                            ExtraParticleData<float> &tree_data,
                            PixelData<float> &temp_vec,
                            py::array &particle_intensities,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half) {

        py::buffer_info particleData = particle_intensities.request(); // pybind11::buffer_info to access data
        auto part_int = (float *) particleData.ptr;

        uint64_t x;

        const uint64_t x_num_m = temp_vec.x_num;
        const uint64_t y_num_m = temp_vec.y_num;



#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

            //
            //  This loop recreates particles at the current level, using a simple copy
            //

            uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);


            //std::cout << "stencil_shape = {" << stencil_shape[0] << ", " << stencil_shape[1] << ", " << stencil_shape[2] << "}" << std::endl;

            for (apr_iterator.set_new_lzx(level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator.global_index()];//particleData.data[apr_iterator];
            }
        }

        if (level > apr_iterator.level_min()) {
            const int y_num = apr_iterator.spatial_index_y_max(level);

            //
            //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
            //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                    temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index()];//particleData[apr_iterator];
                    temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index()];//particleData[apr_iterator];

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_iterator.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(treeIterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (treeIterator.set_new_lzx(level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    temp_vec.at(treeIterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = tree_data[treeIterator];
                }
            }
        }
    }


    template<typename ImageType>
    void update_dense_array2(const uint64_t level,
                             const uint64_t z,
                             APR<ImageType> &apr,
                             APRIterator &apr_iterator,
                             APRTreeIterator &treeIterator,
                             ExtraParticleData<float> &tree_data,
                             PixelData<float> &temp_vec,
                             py::array &particle_intensities,
                             const std::vector<int> &stencil_shape,
                             const std::vector<int> &stencil_half,
                             uint64_t in_offset) {

        py::buffer_info particleData = particle_intensities.request(); // pybind11::buffer_info to access data
        auto part_int = (float *) particleData.ptr;

        uint64_t x;

        const uint64_t x_num_m = temp_vec.x_num;
        const uint64_t y_num_m = temp_vec.y_num;



#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

            //
            //  This loop recreates particles at the current level, using a simple copy
            //

            uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);


            //std::cout << "stencil_shape = {" << stencil_shape[0] << ", " << stencil_shape[1] << ", " << stencil_shape[2] << "}" << std::endl;

            for (apr_iterator.set_new_lzx(level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator.global_index() + in_offset];//particleData.data[apr_iterator];
            }
        }

        if (level > apr_iterator.level_min()) {
            const int y_num = apr_iterator.spatial_index_y_max(level);

            //
            //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
            //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                    temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];
                    temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_iterator.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(treeIterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (treeIterator.set_new_lzx(level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    temp_vec.at(treeIterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = tree_data[treeIterator];
                }
            }
        }
    }


    /**
     * 3^ndim pool downsampling to make an n^ndim stencil (at level l) into a (n-2)^ndim stencil (at level l-1)
     *
     * @tparam T            input data type
     * @tparam S            output data type
     * @tparam R            reduction operator type
     * @tparam C            constant operator type
     * @param input
     * @param output
     * @param reduce
     * @param constant_operator
     * @param aInitializeOutput
     */
    template<typename T, typename S, typename R, typename C>
    void downsample_stencil(PixelData<T> &aInput, PixelData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = true) {

        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        // downsampled dimensions twice smaller (rounded up)
        const size_t z_num_ds = std::max((int)z_num-2, 1);
        const size_t x_num_ds = std::max((int)x_num-2, 1);
        const size_t y_num_ds = std::max((int)y_num-2, 1);

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {

                //const ArrayWrapper<T> &inMesh = aInput.mesh;
                //ArrayWrapper<S> &outMesh = aOutput.mesh;

                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t z = z_ds; z < std::min(z_num, z_ds+3); ++z) {
                        for(size_t x = x_ds; x<std::min(x_num, x_ds+3); ++x) {
                            for(size_t y = y_ds; y<std::min(y_num, y_ds+3); ++y) {
                                outValue = reduce(outValue, aInput.mesh[z*x_num*y_num + x*y_num + y]);
                            }
                        }
                    }

                    aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = constant_operator(outValue);
                }
            }
        }
    }

    /**
     * Downsample a stencil by level_delta levels in such a way that applying the downsampled stencil closely
     * corresponds to applying the original stencil to particles of level = original_level-level_delta.
     * @tparam T                    input data type
     * @tparam S                    output data type
     * @param aInput                input stencil  (PixelData<T>)
     * @param aOutput               output stencil (PixelData<S>)
     * @param level_delta           level difference between input and output
     * @param normalize             should the stencil be normalized (sum to unity)? (default false = no)
     * @param aInitializeOutput     should the output be initialized? (default true = yes)
     */

    template<typename T, typename S>
    void downsample_stencil_alt(const PixelData<T>& aInput, PixelData<S>& aOutput, int level_delta, bool normalize = false, bool aInitializeOutput = true) {

        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        const float size_factor = pow(2, level_delta);
        //const int ndim = (y_num>1) + (x_num > 1) + (z_num>1);

        int k = ceil(z_num / size_factor);
        const size_t z_num_ds = (k % 2 == 0) ? k+1 : k;

        k = ceil(x_num / size_factor);
        const size_t x_num_ds = (k % 2 == 0) ? k+1 : k;

        k = ceil(y_num / size_factor);
        const size_t y_num_ds = (k % 2 == 0) ? k+1 : k;

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        }

        const float offsety = (size_factor*y_num_ds - y_num)/2.0f;
        const float offsetx = (size_factor*x_num_ds - x_num)/2.0f;
        const float offsetz = (size_factor*z_num_ds - z_num)/2.0f;

//#ifdef HAVE_OPENMP
//#pragma omp parallel for default(shared)
//#endif
        float sum = 0;
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {
                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t z = 0; z < z_num; ++z) {
                        for(size_t x = 0; x < x_num; ++x) {
                            for(size_t y = 0; y < y_num; ++y) { // y < std::min((float)y_num, y_ds+size_factor+1) ?

                                float ybegin = y+offsety;
                                float xbegin = x+offsetx;
                                float zbegin = z+offsetz;

                                float overlapy = std::max(size_factor*y_ds, std::min(ybegin+1, size_factor*(y_ds+1))) - std::min(size_factor*(y_ds+1), std::max(ybegin, size_factor*y_ds));
                                float overlapx = std::max(size_factor*x_ds, std::min(xbegin+1, size_factor*(x_ds+1))) - std::min(size_factor*(x_ds+1), std::max(xbegin, size_factor*x_ds));
                                float overlapz = std::max(size_factor*z_ds, std::min(zbegin+1, size_factor*(z_ds+1))) - std::min(size_factor*(z_ds+1), std::max(zbegin, size_factor*z_ds));

                                float factor = overlapy * overlapx * overlapz;

                                outValue += factor * aInput.mesh[z*x_num*y_num + x*y_num + y];
                            }
                        }
                    }

                    aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = outValue; // / pow(size_factor, ndim);
                    sum += outValue;
                }
            }
        }

        if(normalize) {
            float factor = 1.0f / sum;
            for (int i = 0; i < aOutput.mesh.size(); ++i) {
                aOutput.mesh[i] *= factor;
            }
        }/* else {
            float factor = 1.0f / pow(size_factor, ndim);
            for (int i = 0; i < aOutput.mesh.size(); ++i) {
                aOutput.mesh[i] *= factor;
            }
        } */
    }


    template<typename T, typename S>
    void downsample_stencil_cnn2d(const PixelData<T>& aInput, PixelData<S>& aOutput, int level_delta, bool normalize = false, bool aInitializeOutput = true) {

        /// the depth slices of the 3D convolution kernel (corresponding to different input channels) are treated separately
        /// i.e. no averaging over z

        const size_t in_channels = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        const float size_factor = pow(2, level_delta);
        //const int ndim = (y_num>1) + (x_num > 1) + (z_num>1);

        int k = ceil(x_num / size_factor);
        const size_t x_num_ds = (k % 2 == 0) ? k+1 : k;

        k = ceil(y_num / size_factor);
        const size_t y_num_ds = (k % 2 == 0) ? k+1 : k;

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, in_channels);
        }

        const float offsety = (size_factor*y_num_ds - y_num)/2.0f;
        const float offsetx = (size_factor*x_num_ds - x_num)/2.0f;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for (size_t in = 0; in < in_channels; ++in) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {
                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t x = x_ds; x<std::min((float)x_num, x_ds+size_factor+1); ++x) {
                        for(size_t y = y_ds; y<std::min((float)y_num, y_ds+size_factor+1); ++y) {

                            float ybegin = y+offsety;
                            float xbegin = x+offsetx;

                            float overlapy = std::max(size_factor*y_ds, std::min(ybegin+1, size_factor*(y_ds+1))) - std::min(size_factor*(y_ds+1), std::max(ybegin, size_factor*y_ds));
                            float overlapx = std::max(size_factor*x_ds, std::min(xbegin+1, size_factor*(x_ds+1))) - std::min(size_factor*(x_ds+1), std::max(xbegin, size_factor*x_ds));

                            float factor = overlapy * overlapx;

                            outValue += factor * aInput.mesh[in*x_num*y_num + x*y_num + y];
                        }
                    }

                    aOutput.mesh[in*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = outValue; // / pow(size_factor, ndim);
                }
            }
        }

        if(normalize) {
            /// normalize each filter (xy slice) to sum to one?
            int filter_size = x_num_ds*y_num_ds;

            for(int in=0; in < in_channels; ++in) {
                float sum = 0;
                int offset = in*x_num_ds*y_num_ds;

                for(int idx = 0; idx<filter_size; ++idx) {
                    sum += aOutput.mesh[offset+idx];
                }

                float factor = 1.0f / sum;

                for(int idx = 0; idx<filter_size; ++idx) {
                    aOutput.mesh[offset+idx] *= factor;
                }
            }
        }
    }


    template<typename ImageType, typename S>
    void create_test_particles_ds_stencil(APR<ImageType>& apr,
                                          APRIterator& apr_iterator,
                                          APRTreeIterator& apr_tree_iterator,
                                          ExtraParticleData<float>& test_particles,
                                          ExtraParticleData<S>& particles,
                                          ExtraParticleData<float>& part_tree,
                                          PixelData<float> &stencil,
                                          bool normalize=true){

        std::vector<PixelData<float>> stencil_vec;

        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;

        stencil_vec.resize(nstencils);

        PixelData<float> stencilCopy(stencil, true);
        stencil_vec[0].swap(stencilCopy); // first stencil is a copy of the input stencil

        // remaining stencils are downsampled versions of the input stencil
        for(int level_delta = 1; level_delta<nstencils; ++level_delta) {
            downsample_stencil_alt(stencil, stencil_vec[level_delta], level_delta, normalize, true);
        }

        create_test_particles_equiv(apr, apr_iterator, apr_tree_iterator, test_particles, particles, part_tree, stencil_vec);

    }



    template<typename ImageType, typename S>
    void create_test_particles_equiv(APR<ImageType>& apr,
                                     APRIterator& apr_iterator,
                                     APRTreeIterator& apr_tree_iterator,
                                     ExtraParticleData<float>& test_particles,
                                     ExtraParticleData<S>& particles,
                                     ExtraParticleData<float>& part_tree,
                                     const std::vector<PixelData<float>> &stencil_vec){

        int stencil_counter = 0;

        for (uint64_t level_local = apr_iterator.level_max(); level_local >= apr_iterator.level_min(); --level_local) {

            PixelData<float> by_level_recon;
            by_level_recon.init(apr_iterator.spatial_index_y_max(level_local),apr_iterator.spatial_index_x_max(level_local),apr_iterator.spatial_index_z_max(level_local),0);

            //for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {
            for (uint64_t level = apr_iterator.level_min(); level <= level_local; ++level) {
                int z = 0;
                int x = 0;
                const float step_size = pow(2, level_local - level);


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
                for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                    for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                        for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                             apr_iterator.set_iterator_to_particle_next_particle()) {

                            int dim1 = apr_iterator.y() * step_size;
                            int dim2 = apr_iterator.x() * step_size;
                            int dim3 = apr_iterator.z() * step_size;

                            float temp_int;
                            //add to all the required rays

                            temp_int = particles[apr_iterator];

                            const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                            const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                            const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                            for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                                for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                    for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                        by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                                    }
                                }
                            }
                        }
                    }
                }
            }


            if(level_local < apr_iterator.level_max()){

                uint64_t level = level_local;

                const float step_size = 1;

                int z = 0;
                int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_tree_iterator)
#endif
                for (z = 0; z < apr_tree_iterator.spatial_index_z_max(level); z++) {
                    for (x = 0; x < apr_tree_iterator.spatial_index_x_max(level); ++x) {
                        for (apr_tree_iterator.set_new_lzx(level, z, x);
                             apr_tree_iterator.global_index() < apr_tree_iterator.end_index;
                             apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                            int dim1 = apr_tree_iterator.y() * step_size;
                            int dim2 = apr_tree_iterator.x() * step_size;
                            int dim3 = apr_tree_iterator.z() * step_size;

                            float temp_int;
                            //add to all the required rays

                            temp_int = part_tree[apr_tree_iterator];

                            const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                            const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                            const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                            for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                                for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                    for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                        by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            int x = 0;
            int z = 0;
            uint64_t level = level_local;

            PixelData<float> stencil(stencil_vec[stencil_counter], true);

            //const PixelData<float> &stencil = stencil_vec[stencil_counter];
            std::vector<int> stencil_halves = {((int)stencil.y_num-1)/2, ((int)stencil.x_num-1)/2, ((int)stencil.z_num-1)/2};

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {
                //lastly loop over particle locations and compute filter.
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;
                        int counter = 0;

                        const int k = apr_iterator.y(); // offset to allow for boundary padding
                        const int i = x;

                        for (int l = -stencil_halves[2]; l < stencil_halves[2]+1; ++l) {
                            for (int q = -stencil_halves[1]; q < stencil_halves[1]+1; ++q) {
                                for (int w = -stencil_halves[0]; w < stencil_halves[0]+1; ++w) {

                                    if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                        if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                            if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                                neigh_sum += stencil.mesh[counter] * by_level_recon.at(k + w, i + q, z+l);
                                            }
                                        }
                                    }
                                    counter++;
                                }
                            }
                        }

                        test_particles[apr_iterator] = neigh_sum;//std::roundf(neigh_sum/(1.0f*pow((float)2*stencil_halves[0]+1, apr.apr_access.number_dimensions)));
                    }
                }
            }

            std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
            TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

            stencil_counter = std::min(stencil_counter+1, (int)stencil_vec.size()-1);
        }

        PixelData<float> recon_standard;
        apr.interp_img(recon_standard, test_particles);

    }


    template<typename ImageType>
    void convolve_ds_stencil_loop_backward(APR<ImageType> &apr,
                                           py::array &particle_intensities,
                                           PixelData<float> &inputStencil,
                                           py::array &grad_output,
                                           py::array &grad_input,
                                           py::array &grad_weights,
                                           py::array &grad_bias,
                                           int out_channel,
                                           int in_channel) {

        std::vector<PixelData<float>> stencil_vec;
        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;
        stencil_vec.resize(nstencils);

        /// first stencil (pixel resolution) is used as given
        stencil_vec[0].swap(inputStencil);

        /// stencils at lower levels (resolution) are downsampled versions of the input stencil
        for(int level_delta = 1; level_delta<nstencils; ++level_delta) {
            downsample_stencil_alt(stencil_vec[0], stencil_vec[level_delta], level_delta, false, true);
        }

        /*
        for( int i = 0; i<nstencils; ++i){
            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil_dlevel" + std::to_string(i) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil_vec[i]);
        }
        */

        convolve_equivalent_loop_backward(apr, particle_intensities, stencil_vec, grad_output, grad_input, grad_weights, grad_bias, out_channel, in_channel);

    }


    template<typename ImageType, typename T>
    void convolve_equivalent_loop_backward(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, py::array &grad_output, py::array &grad_input, py::array &grad_weight, py::array &grad_bias, int out_channel, int in_channel) {

        //output_intensities.resize(input_intensities.size());

        py::buffer_info grad_input_buf = grad_input.request();
        uint64_t in_offset = in_channel * grad_input_buf.shape[1];

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset);

        /*** initialize a temporary apr tree for the input gradients ***/
        ExtraParticleData<float> grad_tree_temp;

        grad_tree_temp.init(apr.apr_tree.total_number_parent_cells());
        std::fill(grad_tree_temp.data.begin(), grad_tree_temp.data.end(), 0.0f);


        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info grad_output_buf = grad_output.request();
        auto grad_output_ptr = (float *) grad_output_buf.ptr;

        uint64_t out_offset = out_channel * grad_output_buf.shape[1];
        int stencil_counter = 0;

        float d_bias = 0;

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2, (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;


            PixelData<float> temp_vec;
            temp_vec.init(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

            PixelData<float> temp_vec_di;
            temp_vec_di.init(y_num_m, x_num_m, stencil_shape[2], 0);

            PixelData<float> temp_vec_dw;
            temp_vec_dw.init(stencil_vec[stencil_counter].y_num, stencil_vec[stencil_counter].x_num, stencil_vec[stencil_counter].z_num, 0);

            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array2(level,
                                    padd,
                                    apr,
                                    apr_iterator,
                                    tree_iterator,
                                    tree_data,
                                    temp_vec,
                                    input_intensities,
                                    stencil_shape,
                                    stencil_half,
                                    in_offset);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array2(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        int counter = 0;

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                        d_bias += dO;

                        for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                    //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    temp_vec_di.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]) += dO * stencil_vec[stencil_counter].mesh[counter];

                                    temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]);

                                    counter++;
                                }//w
                            }//q
                        }//l
                    }//y, pixels/columns
                }//x , rows

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                update_dense_array2_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp,
                                             temp_vec_di, grad_input, stencil_shape, stencil_half, in_offset);

            }//z

            /// push temp_vec_dw to grad_weights
            downsample_stencil_alt_backward(temp_vec_dw, grad_weight, stencil_counter, out_channel, in_channel);

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to grad_bias
        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        grad_bias_ptr[out_channel] = d_bias;

        /// push grad_tree_temp to grad_inputs
        fill_tree_mean_py_backward(apr, apr.apr_tree, grad_input, grad_tree_temp, in_offset);
    }

    template<typename T>
    void downsample_stencil_alt_backward(const PixelData<T>& ds_grad, py::array &grad_weights, int level_delta, int out_channel, int in_channel) {

        //std::string fileName = "/Users/joeljonsson/Documents/STUFF/grad_weight_delta" + std::to_string(level_delta) + ".tif";
        //TiffUtils::saveMeshAsTiff(fileName, ds_grad);

        const size_t z_num_ds = ds_grad.z_num;
        const size_t x_num_ds = ds_grad.x_num;
        const size_t y_num_ds = ds_grad.y_num;

        const float size_factor = pow(2, level_delta);
        //const int ndim = (y_num>1) + (x_num > 1) + (z_num>1);


        py::buffer_info grad_weight_buf = grad_weights.request(true);
        auto grad_weight_ptr = (float *) grad_weight_buf.ptr;

        const size_t y_num = grad_weight_buf.shape[2];
        const size_t x_num = grad_weight_buf.shape[3];
        const size_t z_num = 1; //TODO: fix for 3D support

        const uint64_t w_offset = out_channel * grad_weight_buf.shape[1] * y_num * x_num + in_channel * y_num * x_num;

        const float offsety = (size_factor*y_num_ds - y_num)/2.0f;
        const float offsetx = (size_factor*x_num_ds - x_num)/2.0f;
        const float offsetz = (size_factor*z_num_ds - z_num)/2.0f;

//#ifdef HAVE_OPENMP
//#pragma omp parallel for default(shared)
//#endif
        float sum = 0;
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {
                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    /*
                     for(size_t z = z_ds; z < std::min((float)z_num, z_ds+size_factor+1); ++z) {
                        for(size_t x = x_ds; x<std::min((float)x_num, x_ds+size_factor+1); ++x) {
                            for(size_t y = y_ds; y<std::min((float)y_num, y_ds+size_factor+1); ++y) {
                     */

                    for(size_t z = 0; z < z_num; ++z) {
                        for(size_t x = 0; x<x_num; ++x) {
                            for(size_t y = 0; y<y_num; ++y) {

                                float ybegin = y+offsety;
                                float xbegin = x+offsetx;
                                float zbegin = z+offsetz;

                                float overlapy = std::max(size_factor*y_ds, std::min(ybegin+1, size_factor*(y_ds+1))) - std::min(size_factor*(y_ds+1), std::max(ybegin, size_factor*y_ds));
                                float overlapx = std::max(size_factor*x_ds, std::min(xbegin+1, size_factor*(x_ds+1))) - std::min(size_factor*(x_ds+1), std::max(xbegin, size_factor*x_ds));
                                float overlapz = std::max(size_factor*z_ds, std::min(zbegin+1, size_factor*(z_ds+1))) - std::min(size_factor*(z_ds+1), std::max(zbegin, size_factor*z_ds));

                                float factor = overlapy * overlapx * overlapz;

                                //outValue += factor * aInput.mesh[z*x_num*y_num + x*y_num + y];

                                grad_weight_ptr[w_offset + y*x_num + x] += factor * ds_grad.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds];
                            }
                        }
                    }

                    //aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = outValue; // / pow(size_factor, ndim);
                    //sum += outValue;
                }
            }
        }
    }


    template<typename ImageType>
    void update_dense_array2_backward(const uint64_t level,
                                      const uint64_t z,
                                      APR<ImageType> &apr,
                                      APRIterator &apr_iterator,
                                      APRTreeIterator &treeIterator,
                                      ExtraParticleData<float> &grad_tree_data,
                                      PixelData<float> &temp_vec_di,
                                      py::array &grad_input,
                                      const std::vector<int> &stencil_shape,
                                      const std::vector<int> &stencil_half,
                                      uint64_t in_offset) {

        py::buffer_info grad_input_buf = grad_input.request(); // pybind11::buffer_info to access data
        auto grad_input_ptr = (float *) grad_input_buf.ptr;

        uint64_t x;

        const uint64_t x_num_m = temp_vec_di.x_num;
        const uint64_t y_num_m = temp_vec_di.y_num;



#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

            //
            //  This loop recreates particles at the current level, using a simple copy
            //

            uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);


            //std::cout << "stencil_shape = {" << stencil_shape[0] << ", " << stencil_shape[1] << ", " << stencil_shape[2] << "}" << std::endl;

            for (apr_iterator.set_new_lzx(level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                //temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator.global_index() + in_offset];//particleData.data[apr_iterator];

                grad_input_ptr[in_offset + apr_iterator.global_index()] += temp_vec_di.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset];
            }
        }

        if (level > apr_iterator.level_min()) {
            const int y_num = apr_iterator.spatial_index_y_max(level);

            //
            //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
            //

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                    //temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                    //            z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];
                    //temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                    //            z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];

                    grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                            temp_vec_di.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]) +
                            temp_vec_di.at(y_m + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_iterator.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(treeIterator)
#endif
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (treeIterator.set_new_lzx(level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    //temp_vec.at(treeIterator.y() + stencil_half[0], x + stencil_half[1],
                    //            z % stencil_shape[2]) = tree_data[treeIterator];

                    grad_tree_data[treeIterator] += temp_vec_di.at(treeIterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

                }
            }
        }
    }

    /*
    template<typename ImageType>
    std::vector<py::array> convolve_ds_stencil_backward(APR<ImageType> &apr, py::array &grad_output, py::array &input, py::array &inputStencil, py::array bias, bool normalize) {

        std::vector<PixelData<float>> stencil_vec;

        py::buffer_info stencil_buf = inputStencil.request();
        //py::buffer_info intensity_buf = particle_intensities.request();

        auto stenc_ptr = (float *)stencil_buf.ptr;

        int nstencils = apr.apr_access.level_max() - apr.apr_access.level_min() + 1;
        stencil_vec.resize(nstencils);

        PixelData<float> stencilCopy;

        switch ((int)stencil_buf.ndim) {

            case 1:
                stencilCopy.init(stencil_buf.shape[0], 1, 1);
                break;
            case 2:
                stencilCopy.init(stencil_buf.shape[0], stencil_buf.shape[1], 1);
                break;
            case 3:
                stencilCopy.init(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2]);
                break;
            default:
                std::cerr <<"could not correctly read input convolution kernel in call to PyAPRFiltering::convolve_ds_stencil" << std::endl;
        }

        for(ssize_t i=0; i<stencil_buf.size; ++i) {
            stencilCopy.mesh[i] = stenc_ptr[i];
        }

        stencil_vec[0].swap(stencilCopy); // first stencil is a copy of the input stencil

        // remaining stencils are downsampled versions of the input stencil
        for(int level_delta = 1; level_delta<nstencils; ++level_delta) {
            downsample_stencil_alt(stencil_vec[0], stencil_vec[level_delta], level_delta, normalize, true);
        }

        for( int i = 0; i<nstencils; ++i){
            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil_dlevel" + std::to_string(i) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil_vec[i]);
        }

        float b = ((float *)bias.request().ptr)[0];

        auto d_output = py::array_t<float_t>(input.request().size);
        auto d_bias = py::array_t<float_t>(bias.request().size);

        std::vector<PixelData<float>> d_stencil_vec;
        d_stencil_vec.resize(nstencils);

        for(int i=0; i<nstencils; ++i) {
            d_stencil_vec[i].init(stencil_vec[i].y_num, stencil_vec[i].x_num, stencil_vec[i].z_num, 0);
        }
        return convolve_equivalent(apr, grad_output, input, stencil_vec, b);

    }
    */




    template<typename T,typename U>
    void fill_tree_mean_py(APR<T>& apr,APRTree<T>& apr_tree, py::array& particle_data,ExtraParticleData<U>& tree_data, uint64_t in_offset) {

        APRTimer timer;
        timer.verbose_flag = false;

        timer.start_timer("ds-init");
        tree_data.init(apr_tree.total_number_parent_cells());

        APRTreeIterator treeIterator = apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr_tree.tree_iterator();

        APRIterator apr_iterator = apr.iterator();

        int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        py::buffer_info input_buf = particle_data.request();
        auto input_ptr = (float *) input_buf.ptr;

        for (unsigned int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < parentIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            //dealing with boundary conditions
                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                                      (z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() <
                                 apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }


                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    tree_data[parentIterator] =
                                            scale_factor_yxz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                                            tree_data[parentIterator];
                                } else {

                                    tree_data[parentIterator] =
                                            scale_factor_xz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                                            tree_data[parentIterator];
                                }

                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        //then do the rest of the tree where order matters
        for (unsigned int level = treeIterator.level_max(); level > treeIterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d,z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z/2, x/2);

                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != parentIterator.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != parentIterator.spatial_index_z_max(level)) &&
                                      ((z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1)))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != parentIterator.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }

                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    tree_data[parentIterator] = scale_factor_yxz * tree_data[treeIterator] / 8.0f +
                                                                tree_data[parentIterator];
                                } else {
                                    tree_data[parentIterator] = scale_factor_xz * tree_data[treeIterator] / 8.0f +
                                                                tree_data[parentIterator];
                                }

                            }
                        }
                    }
                }
            }
        }
        timer.stop_timer();
    }


    template<typename T,typename U>
    void fill_tree_mean_py_backward(APR<T>& apr, APRTree<T>& apr_tree, py::array& grad_input,ExtraParticleData<U>& grad_tree_temp, uint64_t in_offset) {

        APRTimer timer;
        timer.verbose_flag = false;

        APRTreeIterator treeIterator = apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr_tree.tree_iterator();

        APRIterator apr_iterator = apr.iterator();

        int z_d;
        int x_d;
        timer.start_timer("fill_tree_mean_backwards");

        py::buffer_info grad_input_buf = grad_input.request(true);
        auto grad_input_ptr = (float *) grad_input_buf.ptr;


        /// go through the tree from top (low level) to bottom (high level) and push values downwards
        for (unsigned int level = treeIterator.level_min()+1; level <= treeIterator.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d,z_d) firstprivate(treeIterator, parentIterator)
#endif
            for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < treeIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z/2, x/2);

                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != parentIterator.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != parentIterator.spatial_index_z_max(level)) &&
                                      ((z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1)))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != parentIterator.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }

                            for (treeIterator.set_new_lzx(level, z, x);
                                 treeIterator.global_index() <
                                 treeIterator.end_index; treeIterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != treeIterator.y() / 2) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }

                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    //tree_data[parentIterator] = scale_factor_yxz * tree_data[treeIterator] / 8.0f +
                                    //                            tree_data[parentIterator];

                                    grad_tree_temp[treeIterator] += grad_tree_temp[parentIterator] * scale_factor_yxz / 8.0f;

                                } else {
                                    //tree_data[parentIterator] = scale_factor_xz * tree_data[treeIterator] / 8.0f +
                                    //                            tree_data[parentIterator];

                                    grad_tree_temp[treeIterator] += grad_tree_temp[parentIterator] * scale_factor_xz / 8.0f;
                                }

                            }
                        }
                    }
                }
            }
        }


        for (unsigned int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
            for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
                for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
                    for (x_d = 0; x_d < parentIterator.spatial_index_x_max(level-1); ++x_d) {
                        for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)-1); ++x) {

                            parentIterator.set_new_lzx(level - 1, z / 2, x / 2);

                            //dealing with boundary conditions
                            float scale_factor_xz =
                                    (((2 * parentIterator.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                                      ((x / 2) == (parentIterator.spatial_index_x_max(level - 1) - 1))) +
                                     ((2 * parentIterator.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                                      (z / 2) == (parentIterator.spatial_index_z_max(level - 1) - 1))) * 2;

                            if (scale_factor_xz == 0) {
                                scale_factor_xz = 1;
                            }

                            float scale_factor_yxz = scale_factor_xz;

                            if ((2 * parentIterator.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                                scale_factor_yxz = scale_factor_xz * 2;
                            }


                            for (apr_iterator.set_new_lzx(level, z, x);
                                 apr_iterator.global_index() < apr_iterator.end_index;
                                 apr_iterator.set_iterator_to_particle_next_particle()) {

                                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                                    parentIterator.set_iterator_to_particle_next_particle();
                                }


                                if (parentIterator.y() == (parentIterator.spatial_index_y_max(level - 1) - 1)) {
                                    //tree_data[parentIterator] =
                                    //        scale_factor_yxz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                                    //        tree_data[parentIterator];

                                    grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                                            scale_factor_yxz * grad_tree_temp[parentIterator] / 8.0f;
                                } else {

                                    //tree_data[parentIterator] =
                                    //        scale_factor_xz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                                    //        tree_data[parentIterator];

                                    grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                                            scale_factor_xz * grad_tree_temp[parentIterator] / 8.0f;
                                }

                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();

    }


    template<typename ImageType>
    void max_pool(APR<ImageType> &apr,
                  py::array &input,
                  py::array &output,
                  int out_channel,
                  int in_channel) {

        // basically just code from fill_tree_mean_py
        // just a question of what to read and where to put the result (ordering?)

        // first downsample 'current max level' APR particles
        // then downsample 'current max level' tree particles?

    }
};


#endif //LIBAPR_PYAPRFILTERING_HPP
