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
    void convolve_equivalent_loop(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, float bias, py::array &output, int out_channel, int in_channel, int batch_num, int current_max_level) {

        APRTimer timer(false);

        py::buffer_info input_buf = input_intensities.request();

        //int batch_size = input_buf.shape[0];
        int number_in_channels = input_buf.shape[1];
        int nparticles = input_buf.shape[2];

        uint64_t in_offset = batch_num * number_in_channels * nparticles + in_channel * nparticles;

        /**** initialize and fill the apr tree ****/

        ExtraParticleData<float> tree_data;
        //apr.apr_tree.init(apr);

        //unsigned int current_max_level = find_max_level(apr, input_intensities, false);

        timer.start_timer("fill tree");
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset, current_max_level);
        timer.stop_timer();

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info output_buf = output.request(true);
        auto output_ptr = (float *) output_buf.ptr;

        uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + out_channel * output_buf.shape[2];
        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                                   (stencil_shape[1] - 1) / 2,
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
                    timer.start_timer("update_dense_array");
                    update_dense_array2(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                    timer.stop_timer();
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
                    timer.start_timer("padding");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve apr particles");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;
                        int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) { //3D not yet supported
                        for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                            for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                //counter++;

                                neigh_sum += stencil_vec[stencil_counter].mesh[counter] * temp_vec.at(k+w, i+q, 0);
                                counter++;
                            }
                        }
                        //}

                        if(in_channel == number_in_channels-1) {
                            neigh_sum += bias;
                        }

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator)
#endif
                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            float neigh_sum = 0;
                            int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                            const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            const int i = x + stencil_half[1];

                            //compute the output

                            //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                    //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));

                                    //counter++;

                                    neigh_sum += stencil_vec[stencil_counter].mesh[counter] * temp_vec.at(k+w, i+q, 0);
                                    counter++;
                                }
                            }
                            //}

                            if (in_channel == number_in_channels - 1) {
                                neigh_sum += bias;
                            }

                            output_ptr[out_offset + tree_iterator.global_index() + tree_offset] += neigh_sum;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels
        //return output;
    }


    template<typename ImageType>
    void convolve1x1_loop(APR<ImageType> &apr, py::array &input_intensities, const std::vector<float>& stencil_vec, const float bias, py::array &output, const int out_channel, const int in_channel, const int batch_num, const unsigned int current_max_level) {

        APRTimer timer(false);

        /**** initialize the apr tree ****/
        //timer.start_timer("init tree");
        //apr.apr_tree.init(apr);
        //timer.stop_timer();

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        /*** pointers to python data ***/
        py::buffer_info output_buf = output.request(true);
        auto output_ptr = (float *) output_buf.ptr;

        py::buffer_info input_buf = input_intensities.request();
        auto input_ptr = (float *) input_buf.ptr;

        //int batch_size = input_buf.shape[0];
        const int number_in_channels = input_buf.shape[1];
        const int nparticles = input_buf.shape[2];

        const uint64_t in_offset = batch_num * number_in_channels * nparticles + in_channel * nparticles;
        const uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + out_channel * output_buf.shape[2];

        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            unsigned int z = 0;
            unsigned int x = 0;

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve 1x1 apr particles");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = input_ptr[in_offset + apr_iterator.global_index()] * stencil_vec[stencil_counter];

                        if(in_channel == number_in_channels-1) {
                            neigh_sum += bias;
                        }

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve 1x1 tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator)
#endif
                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            const uint64_t idx = tree_iterator.global_index() + tree_offset;

                            float neigh_sum = input_ptr[in_offset + idx] * stencil_vec[stencil_counter];

                            if (in_channel == number_in_channels - 1) {
                                neigh_sum += bias;
                            }

                            output_ptr[out_offset + idx] += neigh_sum;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels
    }


    template<typename ImageType, typename T>
    void convolve3x3_loop_unrolled(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, float bias, py::array &output, int out_channel, int in_channel, int batch_num, int current_max_level) {

        APRTimer timer(false);

        py::buffer_info input_buf = input_intensities.request();

        //int batch_size = input_buf.shape[0];
        const int number_in_channels = input_buf.shape[1];
        const int nparticles = input_buf.shape[2];

        const uint64_t in_offset = batch_num * number_in_channels * nparticles + in_channel * nparticles;

        /**** initialize and fill the apr tree ****/
        timer.start_timer("init tree");

        ExtraParticleData<float> tree_data;
        //apr.apr_tree.init(apr);

        timer.stop_timer();

        //unsigned int current_max_level = find_max_level(apr, input_intensities, false);

        timer.start_timer("fill tree");
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset, current_max_level);
        timer.stop_timer();

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info output_buf = output.request(true);
        auto output_ptr = (float *) output_buf.ptr;

        const uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + out_channel * output_buf.shape[2];
        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                                   (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            timer.start_timer("init temp vec");

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries

            timer.stop_timer();

            timer.start_timer("update temp vec first ('pad')");
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
            timer.stop_timer();

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    timer.start_timer("update_dense_array");
                    update_dense_array2(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                    timer.stop_timer();
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
                    timer.start_timer("padding");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve apr particles");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = temp_vec.at(apr_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                          temp_vec.at(apr_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                          temp_vec.at(apr_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                          temp_vec.at(apr_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                          temp_vec.at(apr_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                          temp_vec.at(apr_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                          temp_vec.at(apr_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                          temp_vec.at(apr_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                          temp_vec.at(apr_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                        if(in_channel == number_in_channels-1) {
                            neigh_sum += bias;
                        }

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator)
#endif
                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            float neigh_sum = temp_vec.at(tree_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                              temp_vec.at(tree_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                              temp_vec.at(tree_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                              temp_vec.at(tree_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                              temp_vec.at(tree_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                              temp_vec.at(tree_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                              temp_vec.at(tree_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                              temp_vec.at(tree_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                              temp_vec.at(tree_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                            if (in_channel == number_in_channels - 1) {
                                neigh_sum += bias;
                            }

                            output_ptr[out_offset + tree_iterator.global_index() + tree_offset] += neigh_sum;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels
    }


    template<typename ImageType, typename T>
    void convolve3x3_loop_unrolled_alt(APR<ImageType> &apr, float * input_intensities,
                                       const std::vector<PixelData<T>>& stencil_vec, float bias,
                                       int out_channel, int in_channel, int batch_num, int current_max_level,
                                       const uint64_t in_offset, ExtraParticleData<float> &tree_data, APRIterator &apr_iterator,
                                       APRTreeIterator &tree_iterator, const size_t number_in_channels,
                                       const uint64_t out_offset, float * output_ptr) {

        APRTimer timer(false);
        /*
        py::buffer_info input_buf = input_intensities.request();

        //int batch_size = input_buf.shape[0];
        const int number_in_channels = input_buf.shape[1];
        const int nparticles = input_buf.shape[2];

        py::buffer_info output_buf = output.request(true);
        auto output_ptr = (float *) output_buf.ptr;

        const uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + out_channel * output_buf.shape[2];
        */
        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                                   (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            timer.start_timer("init temp vec");

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries

            timer.stop_timer();

            timer.start_timer("update temp vec first ('pad')");
            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array3(level,
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
            timer.stop_timer();

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    timer.start_timer("update_dense_array");
                    update_dense_array3(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                    timer.stop_timer();
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
                    timer.start_timer("padding");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve apr particles");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = temp_vec.at(apr_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                          temp_vec.at(apr_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                          temp_vec.at(apr_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                          temp_vec.at(apr_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                          temp_vec.at(apr_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                          temp_vec.at(apr_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                          temp_vec.at(apr_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                          temp_vec.at(apr_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                          temp_vec.at(apr_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                        if(in_channel == number_in_channels-1) {
                            neigh_sum += bias;
                        }

                        //output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator)
#endif
                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            float neigh_sum = temp_vec.at(tree_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                              temp_vec.at(tree_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                              temp_vec.at(tree_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                              temp_vec.at(tree_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                              temp_vec.at(tree_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                              temp_vec.at(tree_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                              temp_vec.at(tree_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                              temp_vec.at(tree_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                              temp_vec.at(tree_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                            if (in_channel == number_in_channels - 1) {
                                neigh_sum += bias;
                            }

                            //output_ptr[out_offset + tree_iterator.global_index() + tree_offset] += neigh_sum;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels
    }


    template<typename ImageType>
    void convolve_ds_stencil_loop(APR<ImageType> &apr,
                                  py::array &particle_intensities,
                                  PixelData<float> &inputStencil,
                                  float bias,
                                  py::array &output,
                                  int out_channel,
                                  int in_channel,
                                  int batch_num,
                                  int level_delta) {

        unsigned int current_max_level = std::max(apr.level_max() - level_delta, apr.level_min()); //find_max_level(apr, particle_intensities, true);

        std::vector<PixelData<float>> stencil_vec;
        int nstencils = current_max_level - apr.level_min() + 1;
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


        convolve_equivalent_loop(apr, particle_intensities, stencil_vec, bias, output, out_channel, in_channel, batch_num, current_max_level);

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

        //apr.apr_tree.init(apr);
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


    template<typename ImageType>
    void update_dense_array3(const uint64_t level,
                             const uint64_t z,
                             APR<ImageType> &apr,
                             APRIterator &apr_iterator,
                             APRTreeIterator &treeIterator,
                             ExtraParticleData<float> &tree_data,
                             PixelData<float> &temp_vec,
                             float * part_int,
                             const std::vector<int> &stencil_shape,
                             const std::vector<int> &stencil_half,
                             uint64_t in_offset) {

        //py::buffer_info particleData = particle_intensities.request(); // pybind11::buffer_info to access data
        //auto part_int = (float *) particleData.ptr;

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


    template<typename ImageType>
    void update_grad_out_arr(const uint64_t level,
                             const uint64_t z,
                             APR<ImageType> &apr,
                             APRIterator &apr_iterator,
                             APRTreeIterator &tree_iterator,
                             PixelData<float> &temp_vec,
                             py::array &particle_intensities,
                             const std::vector<int> &stencil_shape,
                             const std::vector<int> &stencil_half,
                             uint64_t in_offset,
                             unsigned int current_max_level) {

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

        if(level == current_max_level && current_max_level < apr.level_max()) {

            int64_t tree_offset = compute_tree_offset(apr, level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator)
#endif
            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            for (x = 0; x < tree_iterator.spatial_index_x_max(current_max_level); ++x) {

                uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);

                for (tree_iterator.set_new_lzx(current_max_level, z, x);
                     tree_iterator.global_index() < tree_iterator.end_index;
                     tree_iterator.set_iterator_to_particle_next_particle()) {

                    temp_vec.mesh[tree_iterator.y() + stencil_half[0] + mesh_offset] = part_int[tree_offset + tree_iterator.global_index() + in_offset];//particleData.data[apr_iterator];

                }//y
            }//x
        }//if
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
                                           int in_channel,
                                           int batch_num,
                                           int level_delta) {

        unsigned int current_max_level = std::max(apr.level_max() - level_delta, apr.level_min()); //find_max_level(apr, particle_intensities, true);

        std::vector<PixelData<float>> stencil_vec;
        int nstencils = current_max_level - apr.apr_access.level_min() + 1;
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

        convolve_equivalent_loop_backward(apr, particle_intensities, stencil_vec, grad_output, grad_input, grad_weights, grad_bias, out_channel, in_channel, batch_num, current_max_level, true);

    }


    template<typename ImageType, typename T>
    void convolve_equivalent_loop_backward(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, py::array &grad_output, py::array &grad_input, py::array &grad_weight, py::array &grad_bias, int out_channel, int in_channel, int batch_num, unsigned int current_max_level, const bool ds_stencil) {

        APRTimer timer(false);
        APRTimer t2(false);
        //output_intensities.resize(input_intensities.size());

        py::buffer_info grad_input_buf = grad_input.request();
        py::buffer_info grad_output_buf = grad_output.request();

        const uint64_t out_offset = batch_num * grad_output_buf.shape[1] * grad_output_buf.shape[2] + out_channel * grad_output_buf.shape[2];
        const uint64_t in_offset = batch_num * grad_input_buf.shape[1] * grad_input_buf.shape[2] + in_channel * grad_input_buf.shape[2];

        //unsigned int current_max_level = find_max_level(apr, input_intensities, false);

        /**** initialize and fill the apr tree ****/
        timer.start_timer("init tree");
        apr.apr_tree.init(apr);
        timer.stop_timer();

        timer.start_timer("fill tree");
        ExtraParticleData<float> tree_data;
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset, current_max_level);
        timer.stop_timer();

        /*** initialize a temporary apr tree for the input gradients ***/
        ExtraParticleData<float> grad_tree_temp;

        timer.start_timer("init temp tree data");
        grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);
        timer.stop_timer();

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        auto grad_output_ptr = (float *) grad_output_buf.ptr;

        int stencil_counter = 0;

        const int batch_size = grad_input_buf.shape[0];
        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

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

            timer.start_timer("init temp vecs");
            PixelData<float> temp_vec;
            temp_vec.init(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

            PixelData<float> temp_vec_di;
            temp_vec_di.init(y_num_m, x_num_m, stencil_shape[2], 0);

            PixelData<float> temp_vec_dO;
            temp_vec_dO.init(y_num_m, x_num_m, stencil_shape[2], 0);

            PixelData<float> temp_vec_dw;
            int num_threads = 1;
#ifdef HAVE_OPENMP
            num_threads = omp_get_max_threads();
#endif
            //std::cout << "OMP MAX THREADS: " << num_threads << std::endl;

            temp_vec_dw.init(stencil_shape[0], stencil_shape[1], num_threads, 0);

            timer.stop_timer();

            timer.start_timer("fill temp_vec first ('pad')");
            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array2(level, padd, apr, apr_iterator, tree_iterator, tree_data, temp_vec,
                                    input_intensities, stencil_shape, stencil_half, in_offset);

                update_grad_out_arr(level, padd, apr, apr_iterator, tree_iterator,temp_vec_dO, grad_output,
                                    stencil_shape, stencil_half, out_offset, current_max_level);
            }
            timer.stop_timer();

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    timer.start_timer("update dense array");
                    //update the next z plane for the access
                    update_dense_array2(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                        temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);

                    update_grad_out_arr(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, temp_vec_dO,
                                        grad_output, stencil_shape, stencil_half, out_offset, current_max_level);

                    timer.stop_timer();
                } else {
                    //padding
                    timer.start_timer("pad dense array");

                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);

                        std::fill(temp_vec_dO.mesh.begin() + index + (x + 0) * temp_vec_dO.y_num,
                                  temp_vec_dO.mesh.begin() + index + (x + 1) * temp_vec_dO.y_num, 0);
                    }

                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_dO_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec_dO);

                t2.start_timer("BACKWARD CONVOLUTION APR PARTICLES");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x, z) firstprivate(apr_iterator) reduction(+ : d_bias)
#endif
                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    int thread_id = 0;

#ifdef HAVE_OPENMP
                    thread_id = omp_get_thread_num();
#endif

                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const uint64_t idx = apr_iterator.y() + stencil_half[0] + (apr_iterator.x()+stencil_half[1]) * temp_vec_di.y_num;

                        temp_vec_di.mesh[idx] += temp_vec_dO.at(apr_iterator.y(),   x,   0) * stencil_vec[stencil_counter].mesh[8] +
                                                 temp_vec_dO.at(apr_iterator.y()+1, x,   0) * stencil_vec[stencil_counter].mesh[7] +
                                                 temp_vec_dO.at(apr_iterator.y()+2, x,   0) * stencil_vec[stencil_counter].mesh[6] +
                                                 temp_vec_dO.at(apr_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                                 temp_vec_dO.at(apr_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                                 temp_vec_dO.at(apr_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                                 temp_vec_dO.at(apr_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[2] +
                                                 temp_vec_dO.at(apr_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[1] +
                                                 temp_vec_dO.at(apr_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[0];

                    }

                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        //const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        //const int i = x + stencil_half[1];

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];
                        d_bias += dO;

                        //int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;
                        const int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;

                        temp_vec_dw.mesh[w_offset]   += dO * temp_vec.at(apr_iterator.y(),   x, 0);
                        temp_vec_dw.mesh[w_offset+1] += dO * temp_vec.at(apr_iterator.y()+1, x, 0);
                        temp_vec_dw.mesh[w_offset+2] += dO * temp_vec.at(apr_iterator.y()+2, x, 0);

                        temp_vec_dw.mesh[w_offset+3] += dO * temp_vec.at(apr_iterator.y(),   x+1, 0);
                        temp_vec_dw.mesh[w_offset+4] += dO * temp_vec.at(apr_iterator.y()+1, x+1, 0);
                        temp_vec_dw.mesh[w_offset+5] += dO * temp_vec.at(apr_iterator.y()+2, x+1, 0);

                        temp_vec_dw.mesh[w_offset+6] += dO * temp_vec.at(apr_iterator.y(),   x+2, 0);
                        temp_vec_dw.mesh[w_offset+7] += dO * temp_vec.at(apr_iterator.y()+1, x+2, 0);
                        temp_vec_dw.mesh[w_offset+8] += dO * temp_vec.at(apr_iterator.y()+2, x+2, 0);

                    }//y, pixels/columns
                } //ix

                t2.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    int64_t tree_offset = compute_tree_offset(apr, level, false);

                    t2.start_timer("BACKWARD CONV TREE PARTICLES");

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_iterator) reduction(+ : d_bias)
#endif
                    for(x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

                        int thread_id = 0;

#ifdef HAVE_OPENMP
                        thread_id = omp_get_thread_num();
#endif

                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            uint64_t idx = (tree_iterator.y() + stencil_half[0]) + (tree_iterator.x() + stencil_half[1]) * temp_vec_di.y_num;

                            temp_vec_di.mesh[idx] += temp_vec_dO.at(tree_iterator.y(),   x,   0) * stencil_vec[stencil_counter].mesh[8] +
                                                     temp_vec_dO.at(tree_iterator.y()+1, x,   0) * stencil_vec[stencil_counter].mesh[7] +
                                                     temp_vec_dO.at(tree_iterator.y()+2, x,   0) * stencil_vec[stencil_counter].mesh[6] +
                                                     temp_vec_dO.at(tree_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                                     temp_vec_dO.at(tree_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                                     temp_vec_dO.at(tree_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                                     temp_vec_dO.at(tree_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[2] +
                                                     temp_vec_dO.at(tree_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[1] +
                                                     temp_vec_dO.at(tree_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[0];

                        }

                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            const float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            const int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;
                            //int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                            temp_vec_dw.mesh[w_offset]   += dO * temp_vec.at(tree_iterator.y(),   x, 0);
                            temp_vec_dw.mesh[w_offset+1] += dO * temp_vec.at(tree_iterator.y()+1, x, 0);
                            temp_vec_dw.mesh[w_offset+2] += dO * temp_vec.at(tree_iterator.y()+2, x, 0);

                            temp_vec_dw.mesh[w_offset+3] += dO * temp_vec.at(tree_iterator.y(),   x+1, 0);
                            temp_vec_dw.mesh[w_offset+4] += dO * temp_vec.at(tree_iterator.y()+1, x+1, 0);
                            temp_vec_dw.mesh[w_offset+5] += dO * temp_vec.at(tree_iterator.y()+2, x+1, 0);

                            temp_vec_dw.mesh[w_offset+6] += dO * temp_vec.at(tree_iterator.y(),   x+2, 0);
                            temp_vec_dw.mesh[w_offset+7] += dO * temp_vec.at(tree_iterator.y()+1, x+2, 0);
                            temp_vec_dw.mesh[w_offset+8] += dO * temp_vec.at(tree_iterator.y()+2, x+2, 0);
                        }//y, pixels/columns (tree)
                    } //x

                    t2.stop_timer();

                } //if

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                timer.start_timer("update_dense_array2_backward");
                update_dense_array2_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp,
                                             temp_vec_di, grad_input, stencil_shape, stencil_half, in_offset);

                timer.stop_timer();
            }//z

            // sum up weight gradient contributions if >1 threads were used
            if(temp_vec_dw.z_num > 1) {
                timer.start_timer("reduce temp_vec_dw");

                size_t xnumynum = temp_vec_dw.x_num * temp_vec_dw.y_num;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int ixy = 0; ixy < xnumynum; ++ixy) {
                    float tmp = 0.0f;
                    for(int iz = 1; iz < temp_vec_dw.z_num; ++iz) {
                        tmp += temp_vec_dw.mesh[iz * xnumynum + ixy];
                    }

                    temp_vec_dw.mesh[ixy] += tmp;

                }
                timer.stop_timer();
            }
            /// push temp_vec_dw to grad_weights
            timer.start_timer("fill weight gradient");
            if(ds_stencil) {
                downsample_stencil_alt_backward(temp_vec_dw, grad_weight, stencil_counter, out_channel, in_channel, batch_size);
            } else {
                //std::cout << "fill_stencil_gradient called" << std::endl;

                fill_stencil_gradient(temp_vec_dw, grad_weight, out_channel, in_channel, stencil_counter, batch_size);
            }
            timer.stop_timer();
            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to grad_bias
        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        grad_bias_ptr[out_channel] += d_bias / batch_size;

        /// push grad_tree_temp to grad_inputs
        timer.start_timer("fill_tree_backward");
        fill_tree_mean_py_backward(apr, apr.apr_tree, grad_input, grad_tree_temp, in_offset, current_max_level);
        timer.stop_timer();
    }


    template<typename ImageType>
    void convolve1x1_loop_backward(APR<ImageType> &apr, py::array &input_intensities, const std::vector<float>& stencil_vec, py::array &grad_output, py::array &grad_input, py::array &grad_weight, py::array &grad_bias, const int out_channel, const int in_channel, const int batch_num, const unsigned int current_max_level, const bool ds_stencil) {

        APRTimer timer(false);
        APRTimer t2(false);

        /**** initialize the apr tree ****/
        //apr.apr_tree.init(apr);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info grad_output_buf = grad_output.request();
        py::buffer_info grad_input_buf = grad_input.request(true);
        py::buffer_info input_buf = input_intensities.request();
        py::buffer_info grad_weight_buf = grad_weight.request(true);

        auto grad_output_ptr = (float *) grad_output_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto grad_weight_ptr = (float *) grad_weight_buf.ptr;

        const uint64_t in_offset = batch_num * grad_input_buf.shape[1] * grad_input_buf.shape[2] + in_channel * grad_input_buf.shape[2];
        const uint64_t out_offset = batch_num * grad_output_buf.shape[1] * grad_output_buf.shape[2] + out_channel * grad_output_buf.shape[2];

        int stencil_counter = 0;

        const int batch_size = grad_input_buf.shape[0];

        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            float d_weight = 0;

            unsigned int z = 0;
            unsigned int x = 0;

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_bw_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                t2.start_timer("BACKWARD 1x1 CONV APR PARTICLES");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator) reduction(+ : d_bias, d_weight)
#endif
                for(x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                        d_bias += dO;
                        d_weight += dO * input_ptr[in_offset + apr_iterator.global_index()];

                        grad_input_ptr[in_offset + apr_iterator.global_index()] += dO * stencil_vec[stencil_counter];

                    }//y, pixels/columns
                } //x

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

                    t2.start_timer("BACKWARD CONV TREE PARTICLES");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator) reduction(+ : d_bias, d_weight)
#endif
                    for(x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            const uint64_t idx = tree_iterator.global_index() + tree_offset;

                            const float dO = grad_output_ptr[out_offset + idx];

                            d_bias += dO;
                            d_weight += dO * input_ptr[in_offset + idx];

                            grad_input_ptr[in_offset + idx] += dO * stencil_vec[stencil_counter];

                        }//y, pixels/columns (tree)
                    } //x

                    t2.stop_timer();
                } //if
            }//z

            if( !ds_stencil ) {
                const uint64_t w_idx = out_channel * grad_weight_buf.shape[1] * grad_weight_buf.shape[2] +
                                       in_channel * grad_weight_buf.shape[2] +
                                       stencil_counter;

                grad_weight_ptr[w_idx] += d_weight / batch_size;
            } else {
                std::cerr << "convolve1x1_backward is not implemented for downsample stencil approach" << std::endl;
            }

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to grad_bias
        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        grad_bias_ptr[out_channel] += d_bias / batch_size;
    }


    template<typename ImageType, typename T>
    void convolve3x3_loop_unrolled_backward(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, py::array &grad_output, py::array &grad_input, py::array &grad_weight, py::array &grad_bias, int out_channel, int in_channel, int batch_num, unsigned int current_max_level, const bool ds_stencil) {

        APRTimer timer(false);
        APRTimer t2(false);

        py::buffer_info grad_input_buf = grad_input.request();
        const uint64_t in_offset = batch_num * grad_input_buf.shape[1] * grad_input_buf.shape[2] + in_channel * grad_input_buf.shape[2];

        /**** initialize and fill the apr tree ****/
        //apr.apr_tree.init(apr);
        ExtraParticleData<float> tree_data;
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset, current_max_level);

        /*** initialize a temporary apr tree for the input gradients ***/
        ExtraParticleData<float> grad_tree_temp;

        grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info grad_output_buf = grad_output.request();
        auto grad_output_ptr = (float *) grad_output_buf.ptr;

        const uint64_t out_offset = batch_num * grad_output_buf.shape[1] * grad_output_buf.shape[2] + out_channel * grad_output_buf.shape[2];
        int stencil_counter = 0;

        const int batch_size = grad_input_buf.shape[0];
        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

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
            int num_threads = 1;
#ifdef HAVE_OPENMP
            num_threads = omp_get_max_threads();
#endif
            //std::cout << "OMP MAX THREADS: " << num_threads << std::endl;

            temp_vec_dw.init(stencil_shape[0], stencil_shape[1], num_threads, 0);


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

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_bw_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);


                const int chunk_distance = stencil_shape[1];
                const int number_chunks = apr.spatial_index_x_max(level) / chunk_distance;

                t2.start_timer("BACKWARD CONV APR PARTICLES");
                for(int chunk = 0; chunk < chunk_distance; ++chunk) {
                    int ix;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(ix, x) firstprivate(apr_iterator) reduction(+ : d_bias)
#endif
                    for(ix = 0; ix < number_chunks; ++ix) {

                        int thread_id = 0;

#ifdef HAVE_OPENMP
                        thread_id = omp_get_thread_num();
#endif

                        x = chunk + ix * chunk_distance;

                        for (apr_iterator.set_new_lzx(level, z, x);
                             apr_iterator.global_index() < apr_iterator.end_index;
                             apr_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            //compute the stencil

                            const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                            d_bias += dO;

                            //int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;
                            const int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;

                            temp_vec_di.at(apr_iterator.y(), apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[0];
                            temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[1];
                            temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[2];

                            temp_vec_di.at(apr_iterator.y(), apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[3];
                            temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[4];
                            temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[5];

                            temp_vec_di.at(apr_iterator.y(), apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[6];
                            temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[7];
                            temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[8];


                            temp_vec_dw.mesh[w_offset] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x(), 0);
                            temp_vec_dw.mesh[w_offset+1] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x(), 0);
                            temp_vec_dw.mesh[w_offset+2] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x(), 0);

                            temp_vec_dw.mesh[w_offset+3] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x()+1, 0);
                            temp_vec_dw.mesh[w_offset+4] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x()+1, 0);
                            temp_vec_dw.mesh[w_offset+5] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x()+1, 0);

                            temp_vec_dw.mesh[w_offset+6] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x()+2, 0);
                            temp_vec_dw.mesh[w_offset+7] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x()+2, 0);
                            temp_vec_dw.mesh[w_offset+8] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x()+2, 0);

                        }//y, pixels/columns
                    } //ix
                } //chunk

                for(x = number_chunks*chunk_distance; x<apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        //const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        //const int i = x + stencil_half[1];

                        //compute the stencil

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                        //grad_output_ptr[out_offset + apr_iterator.global_index()] = 0.0f;
                        d_bias += dO;

                        //int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                        temp_vec_di.at(apr_iterator.y(), apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[0];
                        temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[1];
                        temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[2];

                        temp_vec_di.at(apr_iterator.y(), apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[3];
                        temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[4];
                        temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[5];

                        temp_vec_di.at(apr_iterator.y(), apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[6];
                        temp_vec_di.at(apr_iterator.y()+1, apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[7];
                        temp_vec_di.at(apr_iterator.y()+2, apr_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[8];


                        temp_vec_dw.mesh[0] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x(), 0);
                        temp_vec_dw.mesh[1] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x(), 0);
                        temp_vec_dw.mesh[2] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x(), 0);

                        temp_vec_dw.mesh[3] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x()+1, 0);
                        temp_vec_dw.mesh[4] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x()+1, 0);
                        temp_vec_dw.mesh[5] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x()+1, 0);

                        temp_vec_dw.mesh[6] += dO * temp_vec.at(apr_iterator.y(), apr_iterator.x()+2, 0);
                        temp_vec_dw.mesh[7] += dO * temp_vec.at(apr_iterator.y()+1, apr_iterator.x()+2, 0);
                        temp_vec_dw.mesh[8] += dO * temp_vec.at(apr_iterator.y()+2, apr_iterator.x()+2, 0);
                    }//y, pixels/columns
                }
                t2.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    int64_t tree_offset = compute_tree_offset(apr, level, false);

                    t2.start_timer("BACKWARD CONV TREE PARTICLES");
                    for(int chunk = 0; chunk < chunk_distance; ++chunk) {
                        int ix;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(ix, x) firstprivate(tree_iterator) reduction(+ : d_bias)
#endif
                        for(ix = 0; ix < number_chunks; ++ix) {

                            int thread_id = 0;

#ifdef HAVE_OPENMP
                            thread_id = omp_get_thread_num();
#endif

                            x = chunk + ix * chunk_distance;

                            for (tree_iterator.set_new_lzx(level, z, x);
                                 tree_iterator.global_index() < tree_iterator.end_index;
                                 tree_iterator.set_iterator_to_particle_next_particle()) {

                                //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                                //const int i = x + stencil_half[1];


                                float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                                d_bias += dO;

                                int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;
                                //int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                                temp_vec_di.at(tree_iterator.y(), tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[0];
                                temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[1];
                                temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[2];

                                temp_vec_di.at(tree_iterator.y(), tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[3];
                                temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[4];
                                temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[5];

                                temp_vec_di.at(tree_iterator.y(), tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[6];
                                temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[7];
                                temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[8];


                                temp_vec_dw.mesh[w_offset] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x(), 0);
                                temp_vec_dw.mesh[w_offset+1] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x(), 0);
                                temp_vec_dw.mesh[w_offset+2] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x(), 0);

                                temp_vec_dw.mesh[w_offset+3] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x()+1, 0);
                                temp_vec_dw.mesh[w_offset+4] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x()+1, 0);
                                temp_vec_dw.mesh[w_offset+5] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x()+1, 0);

                                temp_vec_dw.mesh[w_offset+6] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x()+2, 0);
                                temp_vec_dw.mesh[w_offset+7] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x()+2, 0);
                                temp_vec_dw.mesh[w_offset+8] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x()+2, 0);
                            }//y, pixels/columns (tree)
                        } //ix
                    } //chunk

                    for(x = number_chunks*chunk_distance; x<apr.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            temp_vec_di.at(tree_iterator.y(), tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[0];
                            temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[1];
                            temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x(), 0) += dO * stencil_vec[stencil_counter].mesh[2];

                            temp_vec_di.at(tree_iterator.y(), tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[3];
                            temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[4];
                            temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x()+1, 0) += dO * stencil_vec[stencil_counter].mesh[5];

                            temp_vec_di.at(tree_iterator.y(), tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[6];
                            temp_vec_di.at(tree_iterator.y()+1, tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[7];
                            temp_vec_di.at(tree_iterator.y()+2, tree_iterator.x()+2, 0) += dO * stencil_vec[stencil_counter].mesh[8];


                            temp_vec_dw.mesh[0] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x(), 0);
                            temp_vec_dw.mesh[1] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x(), 0);
                            temp_vec_dw.mesh[2] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x(), 0);

                            temp_vec_dw.mesh[3] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x()+1, 0);
                            temp_vec_dw.mesh[4] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x()+1, 0);
                            temp_vec_dw.mesh[5] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x()+1, 0);

                            temp_vec_dw.mesh[6] += dO * temp_vec.at(tree_iterator.y(), tree_iterator.x()+2, 0);
                            temp_vec_dw.mesh[7] += dO * temp_vec.at(tree_iterator.y()+1, tree_iterator.x()+2, 0);
                            temp_vec_dw.mesh[8] += dO * temp_vec.at(tree_iterator.y()+2, tree_iterator.x()+2, 0);
                        }//y, pixels/columns (tree)
                    }
                    t2.stop_timer();
                } //if

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                timer.start_timer("update_dense_array2_backward");
                update_dense_array2_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp,
                                             temp_vec_di, grad_input, stencil_shape, stencil_half, in_offset);

                timer.stop_timer();
            }//z

            // sum up weight gradient contributions if >1 threads were used
            if(temp_vec_dw.z_num > 1) {
                timer.start_timer("reduce temp_vec_dw");

                size_t xnumynum = temp_vec_dw.x_num * temp_vec_dw.y_num;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int ixy = 0; ixy < xnumynum; ++ixy) {
                    float tmp = 0.0f;
                    for(int iz = 1; iz < temp_vec_dw.z_num; ++iz) {
                        tmp += temp_vec_dw.mesh[iz * xnumynum + ixy];
                    }

                    temp_vec_dw.mesh[ixy] += tmp;

                }
                timer.stop_timer();
            }
            /// push temp_vec_dw to grad_weights
            timer.start_timer("fill weight gradient");
            if(ds_stencil) {
                downsample_stencil_alt_backward(temp_vec_dw, grad_weight, stencil_counter, out_channel, in_channel, batch_size);
            } else {
                //std::cout << "fill_stencil_gradient called" << std::endl;

                fill_stencil_gradient(temp_vec_dw, grad_weight, out_channel, in_channel, stencil_counter, batch_size);
            }
            timer.stop_timer();
            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to grad_bias
        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        grad_bias_ptr[out_channel] += d_bias / batch_size;

        /// push grad_tree_temp to grad_inputs
        timer.start_timer("fill_tree_backward");
        fill_tree_mean_py_backward(apr, apr.apr_tree, grad_input, grad_tree_temp, in_offset, current_max_level);
        timer.stop_timer();
    }


    template<typename ImageType, typename T>
    void convolve_equivalent_loop_backward_old(APR<ImageType> &apr, py::array &input_intensities, const std::vector<PixelData<T>>& stencil_vec, py::array &grad_output, py::array &grad_input, py::array &grad_weight, py::array &grad_bias, int out_channel, int in_channel, int batch_num, unsigned int current_max_level, const bool ds_stencil) {

        APRTimer timer(false);
        //output_intensities.resize(input_intensities.size());

        py::buffer_info grad_input_buf = grad_input.request();
        uint64_t in_offset = batch_num * grad_input_buf.shape[1] * grad_input_buf.shape[2] + in_channel * grad_input_buf.shape[2];

        //unsigned int current_max_level = find_max_level(apr, input_intensities, false);

        /**** initialize and fill the apr tree ****/
        apr.apr_tree.init(apr);
        ExtraParticleData<float> tree_data;
        fill_tree_mean_py(apr, apr.apr_tree, input_intensities, tree_data, in_offset, current_max_level);

        /*** initialize a temporary apr tree for the input gradients ***/
        ExtraParticleData<float> grad_tree_temp;

        grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        py::buffer_info grad_output_buf = grad_output.request();
        auto grad_output_ptr = (float *) grad_output_buf.ptr;

        uint64_t out_offset = batch_num * grad_output_buf.shape[1] * grad_output_buf.shape[2] + out_channel * grad_output_buf.shape[2];
        int stencil_counter = 0;

        int batch_size = grad_input_buf.shape[0];
        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

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
            int num_threads = 1;
#ifdef HAVE_OPENMP
            num_threads = omp_get_max_threads();
#endif
            //std::cout << "OMP MAX THREADS: " << num_threads << std::endl;

            temp_vec_dw.init(stencil_shape[0], stencil_shape[1], num_threads, 0);


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

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_bw_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);


                const int chunk_distance = stencil_shape[1];
                const int number_chunks = apr.spatial_index_x_max(level) / chunk_distance;

                timer.start_timer("backward conv apr particles");
                for(int chunk = 0; chunk < chunk_distance; ++chunk) {
                    int ix;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(ix, x) firstprivate(apr_iterator) reduction(+ : d_bias)
#endif
                    for(ix = 0; ix < number_chunks; ++ix) {

                        int thread_id = 0;

#ifdef HAVE_OPENMP
                        thread_id = omp_get_thread_num();
#endif

                        x = chunk + ix * chunk_distance;

                        for (apr_iterator.set_new_lzx(level, z, x);
                             apr_iterator.global_index() < apr_iterator.end_index;
                             apr_iterator.set_iterator_to_particle_next_particle()) {

                            const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            const int i = x + stencil_half[1];

                            //compute the stencil

                            float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                            d_bias += dO;

                            int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;
                            int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;

                            //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) { //3D not yet supported
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                    //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    //temp_vec_di.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]) += dO * stencil_vec[stencil_counter].mesh[counter];

                                    //temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]);

                                    temp_vec_di.at(k + w, i + q, 0) += dO * stencil_vec[stencil_counter].mesh[counter];
                                    temp_vec_dw.mesh[counter + w_offset] += dO * temp_vec.at(k + w, i + q, 0);

                                    counter++;
                                }//w
                            }//q
                            //}//l
                        }//y, pixels/columns
                    } //ix
                } //chunk

                for(x = number_chunks*chunk_distance; x<apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const int k = apr_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                        const int i = x + stencil_half[1];

                        //compute the stencil

                        float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                        //grad_output_ptr[out_offset + apr_iterator.global_index()] = 0.0f;
                        d_bias += dO;

                        int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                        //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                        for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                            for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                //temp_vec_di.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]) += dO * stencil_vec[stencil_counter].mesh[counter];

                                //temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]);

                                temp_vec_di.at(k + w, i + q, 0) += dO * stencil_vec[stencil_counter].mesh[counter];
                                temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, 0);

                                counter++;
                            }//w
                        }//q
                        //}//l
                    }//y, pixels/columns
                }
                timer.stop_timer();
/*
                /// Loop over APR particles
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator) reduction(+ : d_bias)
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
*/

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    int64_t tree_offset = compute_tree_offset(apr, level, false);

                    timer.start_timer("backward conv tree particles");
                    for(int chunk = 0; chunk < chunk_distance; ++chunk) {
                        int ix;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(ix, x) firstprivate(tree_iterator) reduction(+ : d_bias)
#endif
                        for(ix = 0; ix < number_chunks; ++ix) {

                            int thread_id = 0;

#ifdef HAVE_OPENMP
                            thread_id = omp_get_thread_num();
#endif

                            x = chunk + ix * chunk_distance;

                            for (tree_iterator.set_new_lzx(level, z, x);
                                 tree_iterator.global_index() < tree_iterator.end_index;
                                 tree_iterator.set_iterator_to_particle_next_particle()) {

                                const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                                const int i = x + stencil_half[1];


                                float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                                d_bias += dO;

                                int w_offset = thread_id * temp_vec_dw.x_num * temp_vec_dw.y_num;
                                int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                                //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                                for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                    for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                        //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                        //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));

                                        temp_vec_di.at(k + w, i + q, 0) += dO * stencil_vec[stencil_counter].mesh[counter];

                                        temp_vec_dw.mesh[counter + w_offset] += dO * temp_vec.at(k + w, i + q, 0);

                                        counter++;
                                    }
                                }
                                //}
                            }//y, pixels/columns (tree)
                        } //ix
                    } //chunk

                    for(x = number_chunks*chunk_distance; x<apr.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            const int i = x + stencil_half[1];


                            float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            int counter = 0; //stencil_vec[stencil_counter].mesh.size() - 1;

                            //for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                            for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                    //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                    //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));

                                    temp_vec_di.at(k + w, i + q, 0) += dO * stencil_vec[stencil_counter].mesh[counter];

                                    temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, 0);

                                    counter++;
                                }
                            }
                            //}
                        }//y, pixels/columns (tree)
                    }
                    timer.stop_timer();

                    /*
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(tree_iterator) reduction(+ : d_bias)
#endif
                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            int counter = 0;

                            const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            const int i = x + stencil_half[1];


                            float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            for (int l = -stencil_half[2]; l < stencil_half[2] + 1; ++l) {
                                for (int q = -stencil_half[1]; q < stencil_half[1] + 1; ++q) {
                                    for (int w = -stencil_half[0]; w < stencil_half[0] + 1; ++w) {
                                        //neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                        //              temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));

                                        temp_vec_di.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]) += dO * stencil_vec[stencil_counter].mesh[counter];

                                        temp_vec_dw.mesh[counter] += dO * temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]);

                                        counter++;
                                    }
                                }
                            }
                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                     */
                } //if

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                timer.start_timer("update_dense_array2_backward");
                update_dense_array2_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp,
                                             temp_vec_di, grad_input, stencil_shape, stencil_half, in_offset);

                timer.stop_timer();
            }//z

            // sum up weight gradient contributions if >1 threads were used
            if(temp_vec_dw.z_num > 1) {
                timer.start_timer("reduce temp_vec_dw");

                size_t xnumynum = temp_vec_dw.x_num * temp_vec_dw.y_num;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int ixy = 0; ixy < xnumynum; ++ixy) {
                    float tmp = 0.0f;
                    for(int iz = 1; iz < temp_vec_dw.z_num; ++iz) {
                        tmp += temp_vec_dw.mesh[iz * xnumynum + ixy];
                    }

                    temp_vec_dw.mesh[ixy] += tmp;

                }
                timer.stop_timer();
            }
            /// push temp_vec_dw to grad_weights
            timer.start_timer("fill weight gradient");
            if(ds_stencil) {
                downsample_stencil_alt_backward(temp_vec_dw, grad_weight, stencil_counter, out_channel, in_channel, batch_size);
            } else {
                //std::cout << "fill_stencil_gradient called" << std::endl;

                fill_stencil_gradient(temp_vec_dw, grad_weight, out_channel, in_channel, stencil_counter, batch_size);
            }
            timer.stop_timer();
            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to grad_bias
        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        grad_bias_ptr[out_channel] += d_bias / batch_size;

        /// push grad_tree_temp to grad_inputs
        timer.start_timer("fill_tree_backward");
        fill_tree_mean_py_backward(apr, apr.apr_tree, grad_input, grad_tree_temp, in_offset, current_max_level);
        timer.stop_timer();
    }


    template<typename T>
    void fill_stencil_gradient(const PixelData<T>& temp_dw, py::array &grad_weights, int out_channel, int in_channel, int stencil_counter, int batch_size) {

        py::buffer_info grad_weight_buf = grad_weights.request(true);
        auto grad_weight_ptr = (float *) grad_weight_buf.ptr;

        const size_t y_num = grad_weight_buf.shape[3];
        const size_t x_num = grad_weight_buf.shape[4];
        //const size_t z_num = 1; //TODO: fix for 3D support

        const uint64_t w_offset = out_channel * grad_weight_buf.shape[1] * grad_weight_buf.shape[2] * y_num * x_num +
                                  in_channel * grad_weight_buf.shape[2] * y_num * x_num +
                                  stencil_counter * y_num * x_num;

        /*
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(size_t z = 0; z < z_num; ++z) {
            for (size_t x = 0; x < x_num; ++x) {
                for (size_t y = 0; y < y_num; ++y) {

                    grad_weight_ptr[w_offset + y*x_num + x] += temp_dw.mesh[z*x_num*y_num + x*y_num + y] / batch_size;

                }
            }
        }
         */
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(size_t ixy = 0; ixy < y_num*x_num; ++ixy) {
            grad_weight_ptr[w_offset + ixy] += temp_dw.mesh[ixy] / batch_size;
        }

    }

    template<typename T>
    void downsample_stencil_alt_backward(const PixelData<T>& ds_grad, py::array &grad_weights, int level_delta, int out_channel, int in_channel, int batch_size) {

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

        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
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

                                grad_weight_ptr[w_offset + y*x_num + x] += factor * ds_grad.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] / batch_size;
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

            uint64_t mesh_offset = x_num_m * y_num_m * (z % stencil_shape[2]) + (x + stencil_half[1]) * y_num_m;


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
    void fill_tree_mean_py_ptr(APR<T>& apr,APRTree<T>& apr_tree, float * input_ptr, ExtraParticleData<U>& tree_data, uint64_t in_offset, unsigned int current_max_level) {

        APRTimer timer(false);

        timer.start_timer("ds-init");

        if( current_max_level < apr.level_max() ) {

            tree_data.init(apr_tree.tree_access.global_index_by_level_and_zx_end[current_max_level].back());

        } else {

            tree_data.init(apr_tree.total_number_parent_cells());
        }

        auto treeIterator = apr_tree.tree_iterator();
        auto parentIterator = apr_tree.tree_iterator();
        auto apr_iterator = apr.iterator();

        //int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        //py::buffer_info input_buf = particle_data.request();
        //auto input_ptr = (float *) input_buf.ptr;

        /// if downsampling has been performed, insert the downsampled values directly
        if(current_max_level < apr.level_max()) {

            int z=0;
            int x;

            const int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(treeIterator)
#endif
            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            for (x = 0; x < treeIterator.spatial_index_x_max(current_max_level); ++x) {
                for (treeIterator.set_new_lzx(current_max_level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    tree_data[treeIterator] = input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                }
            }//x
            //}//z
        }//if

        /// now fill in parent nodes of APR particles
        for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
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
            //}
            //}
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        ///then do the rest of the tree where order matters
        for (unsigned int level = std::min(current_max_level, (unsigned int)treeIterator.level_max()); level > treeIterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
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
            //}
            //}
        }
        timer.stop_timer();
    }


    template<typename T,typename U>
    void fill_tree_mean_py(APR<T>& apr,APRTree<T>& apr_tree, py::array& particle_data, ExtraParticleData<U>& tree_data, uint64_t in_offset, unsigned int current_max_level) {

        APRTimer timer(false);

        timer.start_timer("ds-init");

        if( current_max_level < apr.level_max() ) {

            tree_data.init(apr_tree.tree_access.global_index_by_level_and_zx_end[current_max_level].back());

        } else {

            tree_data.init(apr_tree.total_number_parent_cells());
        }

        auto treeIterator = apr_tree.tree_iterator();
        auto parentIterator = apr_tree.tree_iterator();
        auto apr_iterator = apr.iterator();

        //int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        py::buffer_info input_buf = particle_data.request();
        auto input_ptr = (float *) input_buf.ptr;

        /// if downsampling has been performed, insert the downsampled values directly
        if(current_max_level < apr.level_max()) {

            int z=0;
            int x;

            const int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(treeIterator)
#endif
            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            for (x = 0; x < treeIterator.spatial_index_x_max(current_max_level); ++x) {
                for (treeIterator.set_new_lzx(current_max_level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    tree_data[treeIterator] = input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                }
            }//x
            //}//z
        }//if

        /// now fill in parent nodes of APR particles
        for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
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
                //}
            //}
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        ///then do the rest of the tree where order matters
        for (unsigned int level = std::min(current_max_level, (unsigned int)treeIterator.level_max()); level > treeIterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
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
                //}
            //}
        }
        timer.stop_timer();
    }


    template<typename T,typename U>
    void fill_tree_mean_py_backward(APR<T>& apr, APRTree<T>& apr_tree, py::array& grad_input,ExtraParticleData<U>& grad_tree_temp, uint64_t in_offset, unsigned int current_max_level) {

        APRTimer timer(false);
        APRTimer t2(false);

        APRTreeIterator treeIterator = apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr_tree.tree_iterator();

        APRIterator apr_iterator = apr.iterator();

        //int z_d;
        int x_d;
        timer.start_timer("fill_tree_mean_backwards");

        py::buffer_info grad_input_buf = grad_input.request(true);
        auto grad_input_ptr = (float *) grad_input_buf.ptr;


        /// go through the tree from top (low level) to bottom (high level) and push values downwards
        for (unsigned int level = treeIterator.level_min()+1; level <= std::min(current_max_level, (unsigned int)treeIterator.level_max()); ++level) {
            //z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            t2.start_timer("fill tree backward, first loop");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
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
                         treeIterator.global_index() < treeIterator.end_index;
                         treeIterator.set_iterator_to_particle_next_particle()) {

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
            t2.stop_timer();
                //}
            //}
        }


        for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
            //int z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            t2.start_timer("fill tree backward, second loop");
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
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
            t2.stop_timer();
                //}
            //}
        }

        /// if downsampling has been performed, the downsampled values have to be accessed via the tree iterator
        if(current_max_level < apr.level_max()) {

            int z=0;
            int x;

            int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);

            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            t2.start_timer("fill tree backward, third loop");

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x) firstprivate(treeIterator)
#endif
            for (x = 0; x < treeIterator.spatial_index_x_max(current_max_level); ++x) {
                for (treeIterator.set_new_lzx(current_max_level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    //tree_data[treeIterator] = input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                    grad_input_ptr[in_offset + treeIterator.global_index() + tree_offset_in] += grad_tree_temp[treeIterator];

                }
            }//x
            t2.stop_timer();
            //}//z
        }//if
        timer.stop_timer();

    }


    template<typename ImageType>
    void max_pool(APR<ImageType> &apr,
                  py::array &input,
                  py::array &output,
                  unsigned int current_max_level,
                  int batch_num,
                  py::array &index_arr) {

        py::buffer_info input_buf = input.request();
        py::buffer_info output_buf = output.request(true);
        py::buffer_info index_buf = index_arr.request(true);

        const size_t number_channels = input_buf.shape[1];
        const size_t particles_in = input_buf.shape[2];
        const size_t particles_out = output_buf.shape[2];

        if(number_channels != output_buf.shape[1]){
            std::cerr << "number of input and output channels not equal in call to max_pool" << std::endl;
        }

        const uint64_t in_offset = batch_num * number_channels * particles_in;// + channel * input_buf.shape[2];
        const uint64_t out_offset = batch_num * number_channels * particles_out;// + channel * output_buf.shape[2];

        auto input_ptr = (float *) input_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto index_ptr = (uint64_t *) index_buf.ptr;

        /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged

        APRIterator apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            //for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    for(size_t channel=0; channel<number_channels; ++channel) {
                        uint64_t idx_in = apr_iterator.global_index() + in_offset + channel * particles_in;
                        uint64_t idx_out = apr_iterator.global_index() + out_offset + channel * particles_out;

                        output_ptr[idx_out] = input_ptr[idx_in];

                        index_ptr[idx_out] = idx_in;
                    }
                }
            }
            //}
        }
        /// At the current_max_level we may either have purely APR particles, or a mixture of APR and tree particles
        /// (if a downsampling operation was performed previously).

        /// Downsample the APR particles

        APRTreeIterator treeIterator = apr.apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr.apr_tree.tree_iterator();

        const int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);
        const int64_t tree_offset_out = compute_tree_offset(apr, current_max_level-1, false);

        //int z_d = 0;
        int z = 0;
        int x_d;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
        //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(current_max_level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(current_max_level)-1); ++x) {

                parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                for (apr_iterator.set_new_lzx(current_max_level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    while (parentIterator.y() != (apr_iterator.y() / 2)) {
                        parentIterator.set_iterator_to_particle_next_particle();
                    }

                    for(size_t channel=0; channel<number_channels; ++channel) {

                        uint64_t in_idx = in_offset + channel*particles_in + apr_iterator.global_index();
                        uint64_t out_idx = out_offset + channel*particles_out + parentIterator.global_index() + tree_offset_out;
                        //uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                        //uint64_t in_idx = in_offset + apr_iterator.global_index();
                        float curr = output_ptr[out_idx];
                        float tmp = input_ptr[in_idx];

                        if (tmp > curr) {
                            index_ptr[out_idx] = in_idx;

                            output_ptr[out_idx] = tmp;
                        }
                    }
                    //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                    //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                    //                 input_ptr[in_offset + apr_iterator.global_index()]);

                }
            }
        }
        //}
        //}

        /// Now, if the current_max_level is below the maximum level of the APR, it means that APR particles at levels
        /// >= current_max_level have "graduated" to current_max_level. We read these particles using the TreeIterator
        if( current_max_level < apr.level_max()) {
            //int z_d = 0;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(current_max_level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(current_max_level)-1); ++x) {

                    parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                    for (treeIterator.set_new_lzx(current_max_level, z, x);
                         treeIterator.global_index() < treeIterator.end_index;
                         treeIterator.set_iterator_to_particle_next_particle()) {

                        while (parentIterator.y() != (treeIterator.y() / 2)) {
                            parentIterator.set_iterator_to_particle_next_particle();
                        }

                        for(size_t channel=0; channel<number_channels; ++channel) {
                            uint64_t in_idx = in_offset + channel*particles_in + treeIterator.global_index() + tree_offset_in;
                            uint64_t out_idx = out_offset + channel*particles_out + parentIterator.global_index() + tree_offset_out;
                            //uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                            //uint64_t in_idx = in_offset + treeIterator.global_index() + tree_offset_in;
                            float curr = output_ptr[out_idx];
                            float tmp = input_ptr[in_idx];

                            if (tmp > curr) {
                                index_ptr[out_idx] = in_idx;
                                output_ptr[out_idx] = tmp;
                            }
                        }
                        //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                        //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                        //                 input_ptr[in_offset + treeIterator.global_index() + tree_offset_in]);

                    }
                }
            }
            //}
            //}
        }
    }


    template<typename ImageType>
    void max_pool_loop(APR<ImageType> &apr,
                       py::array &input,
                       py::array &output,
                       int channel,
                       unsigned int current_max_level,
                       int batch_num) {

        py::buffer_info input_buf = input.request();
        py::buffer_info output_buf = output.request(true);

        uint64_t in_offset = batch_num * input_buf.shape[1] * input_buf.shape[2] + channel * input_buf.shape[2];
        uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + channel * output_buf.shape[2];

        auto input_ptr = (float *) input_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;

        /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged

        APRIterator apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            //for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    output_ptr[apr_iterator.global_index() + out_offset] = input_ptr[apr_iterator.global_index() + in_offset];

                }
            }
            //}
        }
        /// At the current_max_level we may either have purely APR particles, or a mixture of APR and tree particles
        /// (if a downsampling operation was performed previously).

        /// Downsample the APR particles

        APRTreeIterator treeIterator = apr.apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr.apr_tree.tree_iterator();

        int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);
        int64_t tree_offset_out = compute_tree_offset(apr, current_max_level-1, false);

        //int z_d = 0;

        int z = 0;
        int x_d;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
        //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
        //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(current_max_level)-1); ++z) {
        //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(current_max_level)-1); ++x) {

                parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                for (apr_iterator.set_new_lzx(current_max_level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    while (parentIterator.y() != (apr_iterator.y() / 2)) {
                        parentIterator.set_iterator_to_particle_next_particle();
                    }

                    output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                            std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                                     input_ptr[in_offset + apr_iterator.global_index()]);

                }
            }
        }
        //}
        //}

        /// Now, if the current_max_level is below the maximum level of the APR, it means that APR particles at levels
        /// >= current_max_level have "graduated" to current_max_level. We read these particles using the TreeIterator
        if( current_max_level < apr.level_max()) {
            //int z_d = 0;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(current_max_level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(current_max_level)-1); ++x) {

                    parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                    for (treeIterator.set_new_lzx(current_max_level, z, x);
                         treeIterator.global_index() < treeIterator.end_index;
                         treeIterator.set_iterator_to_particle_next_particle()) {

                        while (parentIterator.y() != (treeIterator.y() / 2)) {
                            parentIterator.set_iterator_to_particle_next_particle();
                        }

                        output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                                std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                                         input_ptr[in_offset + treeIterator.global_index() + tree_offset_in]);

                    }
                }
            }
            //}
            //}
        }
    }


    template<typename ImageType>
    void max_pool_loop_store_idx(APR<ImageType> &apr,
                                 py::array &input,
                                 py::array &output,
                                 int channel,
                                 unsigned int current_max_level,
                                 int batch_num,
                                 py::array &index_arr) {

        py::buffer_info input_buf = input.request();
        py::buffer_info output_buf = output.request(true);
        py::buffer_info index_buf = index_arr.request(true);

        uint64_t in_offset = batch_num * input_buf.shape[1] * input_buf.shape[2] + channel * input_buf.shape[2];
        uint64_t out_offset = batch_num * output_buf.shape[1] * output_buf.shape[2] + channel * output_buf.shape[2];

        auto input_ptr = (float *) input_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto index_ptr = (uint64_t *) index_buf.ptr;

        /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged

        APRIterator apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
            #pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            //for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    output_ptr[apr_iterator.global_index() + out_offset] = input_ptr[apr_iterator.global_index() + in_offset];

                    index_ptr[apr_iterator.global_index() + out_offset] = apr_iterator.global_index() + in_offset;

                }
            }
            //}
        }
        /// At the current_max_level we may either have purely APR particles, or a mixture of APR and tree particles
        /// (if a downsampling operation was performed previously).

        /// Downsample the APR particles

        APRTreeIterator treeIterator = apr.apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr.apr_tree.tree_iterator();

        int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);
        int64_t tree_offset_out = compute_tree_offset(apr, current_max_level-1, false);

        //int z_d = 0;
        int z = 0;
        int x_d;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(apr_iterator, parentIterator)
#endif
        //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(current_max_level)-1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(current_max_level)-1); ++x) {

                parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                for (apr_iterator.set_new_lzx(current_max_level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    while (parentIterator.y() != (apr_iterator.y() / 2)) {
                        parentIterator.set_iterator_to_particle_next_particle();
                    }

                    uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                    uint64_t in_idx = in_offset + apr_iterator.global_index();
                    float curr = output_ptr[out_idx];
                    float tmp = input_ptr[in_idx];

                    if(tmp > curr) {
                        index_ptr[out_idx] = in_idx;

                        output_ptr[out_idx] = tmp;
                    }

                    //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                    //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                    //                 input_ptr[in_offset + apr_iterator.global_index()]);

                }
            }
        }
            //}
        //}

        /// Now, if the current_max_level is below the maximum level of the APR, it means that APR particles at levels
        /// >= current_max_level have "graduated" to current_max_level. We read these particles using the TreeIterator
        if( current_max_level < apr.level_max()) {
            //int z_d = 0;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
                //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(current_max_level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(current_max_level)-1); ++x) {

                    parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                    for (treeIterator.set_new_lzx(current_max_level, z, x);
                         treeIterator.global_index() < treeIterator.end_index;
                         treeIterator.set_iterator_to_particle_next_particle()) {

                        while (parentIterator.y() != (treeIterator.y() / 2)) {
                            parentIterator.set_iterator_to_particle_next_particle();
                        }

                        uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                        uint64_t in_idx = in_offset + treeIterator.global_index() + tree_offset_in;
                        float curr = output_ptr[out_idx];
                        float tmp = input_ptr[in_idx];

                        if(tmp > curr) {
                            index_ptr[out_idx] = in_idx;
                            output_ptr[out_idx] = tmp;
                        }

                        //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                        //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                        //                 input_ptr[in_offset + treeIterator.global_index() + tree_offset_in]);

                    }
                }
            }
                //}
            //}
        }
    }


    template<typename ImageType>
    void max_pool_loop_backward(APR<ImageType> &apr,
                                py::array &input,
                                py::array &grad_input,
                                py::array &grad_output,
                                int channel,
                                unsigned int current_max_level,
                                int batch_num) {

        py::buffer_info input_buf = input.request();
        py::buffer_info grad_input_buf = grad_input.request();
        py::buffer_info grad_output_buf = grad_output.request(true);

        uint64_t in_offset = batch_num * grad_input_buf.shape[1] * grad_input_buf.shape[2] + channel * grad_input_buf.shape[1];
        uint64_t out_offset = batch_num * grad_output_buf.shape[1] * grad_output_buf.shape[2] + channel * grad_output_buf.shape[1];

        auto input_ptr = (float *) input_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto grad_output_ptr = (float *) grad_output_buf.ptr;

        /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged

        APRIterator apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            //for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    grad_input_ptr[apr_iterator.global_index() + in_offset] = grad_output_ptr[apr_iterator.global_index() + out_offset];

                    //temp_vec[apr_iterator.global_index()] = input_ptr[apr_iterator.global_index() + in_offset];

                }
            }
            //}
        }
        /// At the current_max_level we may either have purely APR particles, or a mixture of APR and tree particles
        /// (if a downsampling operation was performed previously).

        /// backwards downsample the APR particles

        std::vector<float> temp_vec; // temporary vector in which to store the maximum values
        uint64_t number_ds_outputs = grad_output_buf.shape[2];// - apr.apr_access.global_index_by_level_and_zx_end[current_max_level - 1].back();
        temp_vec.resize(number_ds_outputs, -600000.0f);

        std::vector<int64_t> index_vec;
        index_vec.resize(number_ds_outputs, -1);

        APRTreeIterator treeIterator = apr.apr_tree.tree_iterator();
        APRTreeIterator parentIterator = apr.apr_tree.tree_iterator();

        int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);
        int64_t tree_offset_out = compute_tree_offset(apr, current_max_level-1, false);

        int z_d = 0;
        int z = 0;
        int x_d;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(apr_iterator, parentIterator)
#endif
        //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(current_max_level)-1); ++z) {
                //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
        for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
            for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(current_max_level)-1); ++x) {

                parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                for (apr_iterator.set_new_lzx(current_max_level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    while (parentIterator.y() != (apr_iterator.y() / 2)) {
                        parentIterator.set_iterator_to_particle_next_particle();
                    }

                    if( input_ptr[in_offset + apr_iterator.global_index()] > temp_vec[parentIterator.global_index() + tree_offset_out] ) {
                        //grad_input_ptr[in_offset + apr_iterator.global_index()] =
                        //        grad_output_ptr[out_offset + parentIterator.global_index() + tree_offset_out];

                        index_vec[parentIterator.global_index() + tree_offset_out] = in_offset+apr_iterator.global_index();
                        temp_vec[parentIterator.global_index() + tree_offset_out] = input_ptr[in_offset + apr_iterator.global_index()];
                    }

                    //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                    //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                    //                 input_ptr[in_offset + apr_iterator.global_index()]);

                }
            }
        }
            //}
        //}

        /// Now, if the current_max_level is below the maximum level of the APR, it means that APR particles at levels
        /// >= current_max_level have "graduated" to current_max_level. We read these particles using the TreeIterator
        if( current_max_level < apr.level_max()) {
            z_d = 0;
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x_d, z_d) firstprivate(treeIterator, parentIterator)
#endif
            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(current_max_level-1); z_d++) {
                //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(current_max_level)-1); ++z) {
                    //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < parentIterator.spatial_index_x_max(current_max_level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) treeIterator.spatial_index_x_max(current_max_level)-1); ++x) {

                    parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                    for (treeIterator.set_new_lzx(current_max_level, z, x);
                         treeIterator.global_index() < treeIterator.end_index;
                         treeIterator.set_iterator_to_particle_next_particle()) {

                        while (parentIterator.y() != (treeIterator.y() / 2)) {
                            parentIterator.set_iterator_to_particle_next_particle();
                        }

                        if(input_ptr[in_offset + treeIterator.global_index() + tree_offset_in] > temp_vec[parentIterator.global_index() + tree_offset_out]) {

                            //grad_input_ptr[in_offset + treeIterator.global_index() + tree_offset_in] =
                            //        grad_output_ptr[out_offset + parentIterator.global_index() + tree_offset_out];

                            index_vec[parentIterator.global_index() + tree_offset_out] = in_offset + treeIterator.global_index() + tree_offset_in;

                            temp_vec[parentIterator.global_index() + tree_offset_out] =
                                    input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                        }

                        //output_ptr[out_offset + parentIterator.global_index() + tree_offset_out] =
                        //        std::max(output_ptr[out_offset + parentIterator.global_index() + tree_offset_out],
                        //                 input_ptr[in_offset + treeIterator.global_index() + tree_offset_in]);

                    }
                }//x
            }//x_d
                //}//z
            //}//z_d
        }//if
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int idx = 0; idx < index_vec.size(); ++idx) {
            if(index_vec[idx] >= 0) {
                grad_input_ptr[index_vec[idx]] = grad_output_ptr[out_offset + idx];
            }
        }
    }


    template<typename ImageType>
    unsigned int find_max_level(APR<ImageType> &apr, py::array &input_features, bool init_tree = true) {

        py::buffer_info input_buf = input_features.request();

        /// Find the current maximum level using the shape of the input
        uint64_t unknown_size = input_buf.shape[2];


        if(init_tree) { apr.apr_tree.init(apr); }

        unsigned int find_level = apr.level_max();
        uint64_t parts_guess = apr.total_number_particles();

        while((unknown_size < parts_guess) && (find_level > apr.level_min()) ) {
            find_level--;

            unsigned int number_parts = apr.apr_access.global_index_by_level_and_zx_end[find_level].back();

            unsigned int tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[find_level - 1].back();
            unsigned int tree_end = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[find_level].back();

            unsigned int number_graduated_parts = tree_end - tree_start;

            parts_guess = number_parts + number_graduated_parts;
        }

        return find_level;

    }

    template<typename ImageType>
    int64_t compute_tree_offset(APR<ImageType> &apr, unsigned int level, bool init_tree = false) {

        if(init_tree) { apr.apr_tree.init(apr); }

        int number_parts;
        int tree_start;

        if(level >= apr.level_min()) {
            number_parts = apr.apr_access.global_index_by_level_and_zx_end[level].back();
        } else {
            number_parts = 0;
        }

        if(level > apr.apr_tree.tree_access.level_min()) {
            tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[level - 1].back();
        } else {
            tree_start = 0;
        }

        int64_t tree_offset = number_parts - tree_start;

        return tree_offset;
    }


    /**
     * Compute a piecewise constant reconstruction using the provided py::array of particle intensities, that may
     * include downsampled values that have to be read using the apr tree
     */
    template<typename U,typename S>
    void interp_img_new_intensities(APR<S>& apr, PixelData<U>& img, py::array &intensities, int level_delta){

        //  Takes in an APR and a python array of intensities to create piece-wise constant image

        py::buffer_info buf = intensities.request();
        auto intptr = (float *) buf.ptr;


        //unsigned int current_max_level = find_max_level(apr, intensities, true);
        unsigned int current_max_level = std::max(apr.level_max()-level_delta, apr.level_min());

        img.init(apr.apr_access.y_num[current_max_level], apr.apr_access.x_num[current_max_level], apr.apr_access.z_num[current_max_level], 0);

        APRIterator apr_iterator = apr.iterator();

        //int max_dim = std::max(std::max(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]), apr.apr_access.org_dims[2]);
        //int max_level = ceil(std::log2(max_dim));

        /// fill in values from the APR
        for (unsigned int level = apr_iterator.level_min(); level <= current_max_level; ++level) {
            int z = 0;
            int x = 0;

            const float step_size = pow(2, current_max_level - level);

#ifdef HAVE_OPENMP
            const bool parallel_z = apr_iterator.spatial_index_z_max(level) > 1;
            const bool parallel_x = !parallel_z && apr_iterator.spatial_index_x_max(level) > 1;
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) if(parallel_z)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) if(parallel_x)
#endif
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {
                        //
                        //  Parallel loop over level
                        //

                        int dim1 = apr_iterator.y() * step_size;
                        int dim2 = apr_iterator.x() * step_size;
                        int dim3 = apr_iterator.z() * step_size;

                        float temp_int;
                        //add to all the required rays

                        temp_int = intptr[apr_iterator.global_index()];

                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        for (int64_t q = dim3; q < offset_max_dim3; ++q) {
                            for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                    img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = temp_int;
                                }
                            }
                        }
                    }
                }
            }
        }

        /// fill in eventual downsampled values using the tree iterator
        if(current_max_level < apr.level_max()) {

            APRTreeIterator tree_iterator = apr.apr_tree.tree_iterator();

            int64_t tree_offset = compute_tree_offset(apr, current_max_level, false);

            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
            const bool parallel_z = tree_iterator.spatial_index_z_max(current_max_level) > 1000;
            const bool parallel_x = !parallel_z && tree_iterator.spatial_index_x_max(current_max_level) > 1000;
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_iterator) if(parallel_z)
#endif
            for (z = 0; z < tree_iterator.spatial_index_z_max(current_max_level); z++) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_iterator) if(parallel_x)
#endif
                for (x = 0; x < tree_iterator.spatial_index_x_max(current_max_level); ++x) {
                    for (tree_iterator.set_new_lzx(current_max_level, z, x);
                         tree_iterator.global_index() < tree_iterator.end_index;
                         tree_iterator.set_iterator_to_particle_next_particle()) {

                        int dim1 = tree_iterator.y();
                        int dim2 = tree_iterator.x();
                        int dim3 = tree_iterator.z();

                        img.mesh[dim3*img.x_num*img.y_num + dim2*img.y_num + dim1] = intptr[tree_offset + tree_iterator.global_index()];

                    }
                }
            }
        }
    }


    /**
     * Computes the required number of intensity values required to represent the image with a given maximum level.
     */
    template<typename ImageType>
    uint64_t number_parts_at_level(APR<ImageType> &apr, unsigned int max_level) {

        unsigned int number_parts;
        unsigned int tree_start;
        unsigned int tree_end;

        if(max_level >= apr.level_min()) {
            number_parts = apr.apr_access.global_index_by_level_and_zx_end[max_level].back();
        } else {
            number_parts = 0;
        }

        if(max_level > apr.apr_tree.tree_access.level_min()) {
            tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level - 1].back();
        } else {
            tree_start = 0;
        }

        if(max_level >= apr.apr_tree.tree_access.level_min()) {
            tree_end = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level].back();
        } else {
            tree_end = 0;
        }

        unsigned int number_graduated_parts = tree_end - tree_start;

        uint64_t number_parts_out = number_parts + number_graduated_parts;

        return number_parts_out;
    }



};


#endif //LIBAPR_PYAPRFILTERING_HPP
