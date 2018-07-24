//
// Created by Joel Jonsson on 18.07.18.
//

#ifndef LIBAPR_APRFILTERING_HPP
#define LIBAPR_APRFILTERING_HPP

#include <iostream>
#include <vector>

#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
//#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"

class APRFiltering {

public:

    // TODO: should it directly change the internal apr intensities?
    template<typename ImageType, typename S, typename T>
    void convolve_equivalent(APR<ImageType> &apr, const std::vector<PixelData<T>>& stencil_vec, ExtraParticleData<S> &particle_intensities, ExtraParticleData<float> &conv_particle_intensities) {

        conv_particle_intensities.init(particle_intensities.total_number_particles());

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, particle_intensities, tree_data);

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
                                    neigh_sum += (stencil_vec[stencil_counter].mesh[counter] *
                                                  temp_vec.at(k + w, i + q, (z + stencil_half[2] + l) % stencil_shape[2]));
                                    counter++;
                                }
                            }
                        }

                        conv_particle_intensities[apr_iterator] = neigh_sum;//std::roundf(neigh_sum/(norm*1.0f));

                    }//y, pixels/columns
                }//x , rows
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels
    }


    template<typename ImageType, typename S>
    void convolve_ds_stencil(APR<ImageType> &apr,
                             PixelData<float> &inputStencil,
                             ExtraParticleData<S> &particle_intensities,
                             ExtraParticleData<float> &conv_particle_intensities) {

        conv_particle_intensities.init(particle_intensities.total_number_particles());

        /**** initialize and fill the apr tree ****/
        ExtraParticleData<float> tree_data;

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, particle_intensities, tree_data);

        /*** iterators for accessing apr data ***/
        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();

        PixelData<float> stencil(inputStencil, true);

        for (int level = apr_iterator.level_max(); level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil.y_num,
                                                    (int) stencil.x_num,
                                                    (int) stencil.z_num};

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

                        conv_particle_intensities[apr_iterator] = neigh_sum;//std::roundf(neigh_sum/(norm*1.0f));

                    }//y, pixels/columns
                }//x , rows
            }//z

            // downsample the stencil for the next level

            std::string fileName = "/Users/joeljonsson/Documents/STUFF/stencil" + std::to_string(level) + ".tif";
            TiffUtils::saveMeshAsTiff(fileName, stencil);

            if(stencil.mesh.size() > 1) {
                PixelData<float> stencil_ds;

                downsample_stencil_alt(stencil, stencil_ds,
                                       [](const float &x, const float &y) -> float { return x + y; },
                                       [&apr](const float &x) -> float { return x / pow(3, apr.apr_access.number_dimensions); }, true);

                std::swap(stencil, stencil_ds);
            }

        }//levels
    }


    template<typename ImageType>
    void update_dense_array(const uint64_t level,
                            const uint64_t z,
                            APR<ImageType> &apr,
                            APRIterator &apr_iterator,
                            APRTreeIterator &treeIterator,
                            ExtraParticleData<float> &tree_data,
                            PixelData<float> &temp_vec,
                            ExtraParticleData<ImageType> &particleData,
                            const std::vector<int> &stencil_shape,
                            const std::vector<int> &stencil_half) {

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

                temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = particleData.data[apr_iterator];
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
                                z % stencil_shape[2]) = particleData[apr_iterator];
                    temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = particleData[apr_iterator];

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


    template<typename T, typename S, typename R, typename C>
    void downsample_stencil_alt(PixelData<T> &aInput, PixelData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = true) {

        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        size_t y_num_ds;
        int k = (y_num+1)/2;
        if( k > 1) {
            if (k % 2 == 0) {
                int lambda = (2 * (k + 1) - y_num) < (y_num - 2 * (k - 1)) ? 1 : -1;
                y_num_ds = std::max(k + lambda, 1);
            } else {
                y_num_ds = k;
            }
        } else {
            y_num_ds = 1;
        }

        size_t x_num_ds;
        k = (x_num+1)/2;
        if( k > 1) {
            if (k % 2 == 0) {
                int lambda = (2*(k+1) - x_num) < (x_num - 2*(k-1)) ? 1 : -1;
                x_num_ds = std::max(k + lambda, 1);
            } else {
                x_num_ds = k;
            }
        } else {
            x_num_ds = 1;
        }

        size_t z_num_ds;
        k = (z_num+1)/2;
        if( k > 1) {
            if (k % 2 == 0) {
                int lambda = (2*(k+1) - z_num) < (z_num - 2*(k-1)) ? 1 : -1;
                z_num_ds = std::max(k + lambda, 1);
            } else {
                z_num_ds = k;
            }
        } else {
            z_num_ds = 1;
        }


        //const size_t y_num_ds = y_num > 1 ? (y_num+1)/2 + (1-((y_num+1)/2)%2) : 1;
        //const size_t x_num_ds = x_num > 1 ? (x_num+1)/2 + (1-((x_num+1)/2)%2) : 1;
        //const size_t z_num_ds = z_num > 1 ? (z_num+1)/2 + (1-((z_num+1)/2)%2) : 1;

        if (aInitializeOutput) {
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
        }

        const float offsety = (2.0f*y_num_ds - y_num)/2.0f;
        const float offsetx = (2.0f*x_num_ds - x_num)/2.0f;
        const float offsetz = (2.0f*z_num_ds - z_num)/2.0f;

//#ifdef HAVE_OPENMP
//#pragma omp parallel for default(shared)
//#endif
        for (size_t z_ds = 0; z_ds < z_num_ds; ++z_ds) {
            for (size_t x_ds = 0; x_ds < x_num_ds; ++x_ds) {
                for (size_t y_ds = 0; y_ds < y_num_ds; ++y_ds) {

                    float outValue = 0;

                    for(size_t z = z_ds; z < std::min(z_num, z_ds+3); ++z) {
                        for(size_t x = x_ds; x<std::min(x_num, x_ds+3); ++x) {
                            for(size_t y = y_ds; y<std::min(y_num, y_ds+3); ++y) {

                                float ybegin = y+offsety;
                                float xbegin = x+offsetx;
                                float zbegin = z+offsetz;

                                float overlapy = std::max(2.0f*y_ds, std::min(ybegin+1, 2.0f*y_ds+2)) - std::min(2.0f*y_ds+2, std::max(ybegin, 2.0f*y_ds));
                                float overlapx = std::max(2.0f*x_ds, std::min(xbegin+1, 2.0f*x_ds+2)) - std::min(2.0f*x_ds+2, std::max(xbegin, 2.0f*x_ds));
                                float overlapz = std::max(2.0f*z_ds, std::min(zbegin+1, 2.0f*z_ds+2)) - std::min(2.0f*z_ds+2, std::max(zbegin, 2.0f*z_ds));

                                float factor = overlapy * overlapx * overlapz;

                                outValue = reduce(outValue, factor * aInput.mesh[z*x_num*y_num + x*y_num + y]);
                            }
                        }
                    }

                    aOutput.mesh[z_ds*x_num_ds*y_num_ds + x_ds*y_num_ds + y_ds] = constant_operator(outValue);
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
                                          PixelData<float> &stencil){


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

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);
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

            //std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
            //TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

            if(stencil.mesh.size() > 1) {

                PixelData<float> stencil_ds;

                downsample_stencil_alt(stencil, stencil_ds,
                                       [](const float &x, const float &y) -> float { return x + y; },
                                       [&apr](const float &x) -> float { return x / pow(3, apr.apr_access.number_dimensions); }, true);

                std::swap(stencil, stencil_ds);
            }

        }

        PixelData<float> recon_standard;
        apr.interp_img(recon_standard, test_particles);

        TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/conv_recon_standard.tif",recon_standard);

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

        TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/conv_recon_standard.tif",recon_standard);

    }


};

#endif //LIBAPR_APRFILTERING_HPP
