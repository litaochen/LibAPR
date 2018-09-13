//
// Created by cheesema on 28.02.18.
//

//
// Created by cheesema on 28.02.18.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(


)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"

#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include <numerics/APRNumerics.hpp>
//#include <numerics/APRComputeHelper.hpp>

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    int num_rep = 10;
    int stencil_size = 1;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

void create_test_particles(APR<uint16_t>& apr,APRIterator& apr_iterator,APRTreeIterator& apr_tree_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<float>& part_tree,std::vector<float>& stencil, const int stencil_size, const int stencil_half);

template<typename T,typename ParticleDataType>
void update_dense_array(const uint64_t level,const uint64_t z,APR<uint16_t>& apr,APRIterator& apr_iterator, APRTreeIterator& treeIterator, ExtraParticleData<float> &tree_data,PixelData<T>& temp_vec,ExtraParticleData<ParticleDataType>& particleData, const int stencil_size, const int stencil_half) {

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

        uint64_t mesh_offset = (x + stencil_half) * y_num_m + x_num_m * y_num_m * (z % stencil_size);


        for(apr_iterator.set_new_lzx(level, z, x);
            apr_iterator.global_index() < apr_iterator.end_index;
            apr_iterator.set_iterator_to_particle_next_particle()) {

            temp_vec.mesh[apr_iterator.y() + stencil_half + mesh_offset] = particleData.data[apr_iterator];
        }

        /*
        apr_iterator.set_new_lzx(level, z, x);
        for (unsigned long gap = 0;
             gap < apr_iterator.number_gaps(); apr_iterator.move_gap(gap)) {

            uint64_t y_begin = apr_iterator.current_gap_y_begin() ;
            uint64_t y_end = apr_iterator.current_gap_y_end() ;
            uint64_t index = apr_iterator.current_gap_index();

            std::copy(particleData.data.begin() + index, particleData.data.begin() + index + (y_end - y_begin) +1,
                      temp_vec.mesh.begin() + mesh_offset + y_begin + stencil_half);


        }
        */

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

                int y_m = std::min(2 * apr_iterator.y() + 1, y_num-1);	// 2y+1+offset

                temp_vec.at(2 * apr_iterator.y() + stencil_half, x + stencil_half, z % stencil_size) = particleData[apr_iterator];
                temp_vec.at(y_m + stencil_half, x + stencil_half, z % stencil_size) = particleData[apr_iterator];

            }

        }
    }

    /******** start of using the tree iterator for downsampling ************/


// #TODO OpenMP?
    if (level < apr_iterator.level_max()) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(treeIterator)
#endif
        for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
            for (treeIterator.set_new_lzx(level, z , x );
                 treeIterator.global_index() < treeIterator.end_index;
                 treeIterator.set_iterator_to_particle_next_particle()) {

                temp_vec.at(treeIterator.y() + stencil_half, x +stencil_half, z % stencil_size) = tree_data[treeIterator];
            }
        }
    }
}
		




int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    int num_rep = options.num_rep;

    timer.start_timer("APR serial iterator neighbours loop");

    //Basic serial iteration over all particles

  	/**** Filling the inside ********/

    timer.start_timer("filling the tree");

    for (int m = 0; m < num_rep ; ++m) {

        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean_downsample(apr.particles_intensities);

    }

    timer.stop_timer();
    float ds_time = timer.timings.back();

	// treeIterator only stores the Inside of the tree
   /****** End of filling **********/

    ExtraParticleData<float> part_sum(apr.total_number_particles());

    std::cout << 1000000.0*1000*ds_time/(num_rep*(1.0f)*apr.total_number_particles())  << " ms million DS Tree" << std::endl;
    const int stencil_half = options.stencil_size;
    const int stencil_size = 2*stencil_half + 1;

    std::vector<float>  stencil;

    float norm = pow(stencil_size,3);

    float stencil_value = 1.0f/norm;

    stencil.resize(norm, stencil_value);

    ExtraParticleData<float> part_sum_dense(apr.total_number_particles());

    timer.start_timer("Dense neighbour access");
    ExtraParticleData<float> tree_data;

    for (int j = 0; j < num_rep; ++j) {

        // fill the apr tree
        apr.apr_tree.init(apr);
        apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, apr.particles_intensities, tree_data);


        auto apr_iterator = apr.iterator();
        auto tree_iterator = apr.apr_tree.tree_iterator();


        for (int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

            unsigned int z = 0;
            unsigned int x = 0;

            const int y_num = apr_iterator.spatial_index_y_max(level);
            const int x_num = apr_iterator.spatial_index_x_max(level);
            const int z_num = apr_iterator.spatial_index_z_max(level);

            PixelData<float> temp_vec;
            temp_vec.init(y_num + (stencil_size-1),
                          x_num + (stencil_size-1),
                          stencil_size,
                          0); //padded boundaries

            z = 0;

            //initial condition
            for (int padd = 0; padd < stencil_half; ++padd) {
                update_dense_array(level,
                                   padd,
                                   apr,
                                   apr_iterator,
                                   tree_iterator,
                                   tree_data,
                                   temp_vec,
                                   apr.particles_intensities,
                                   stencil_size,
                                   stencil_half);
            }

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - (stencil_half))) {
                    //update the next z plane for the access
                    update_dense_array(level, z + stencil_half, apr, apr_iterator, tree_iterator, tree_data, temp_vec,apr.particles_intensities, stencil_size, stencil_half);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z+stencil_half)%stencil_size);
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(x)
#endif
                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num ,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num , 0);
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

                        const int k = apr_iterator.y() + stencil_half; // offset to allow for boundary padding
                        const int i = x + stencil_half;

                        //compute the stencil
                        for (int l = -stencil_half; l < stencil_half+1; ++l) {
                            for (int q = -stencil_half; q < stencil_half+1; ++q) {
                                for (int w = -stencil_half; w < stencil_half+1; ++w) {
                                    neigh_sum += (stencil[counter]*temp_vec.at(k+w, i+q, (z+stencil_size+l)%stencil_size));
                                    counter++;
                                }
                            }
                        }

                        part_sum_dense[apr_iterator] = std::roundf(neigh_sum/(norm*1.0f));

                    }//y, pixels/columns
                }//x , rows
            }//z
        }//levels
    }//reps


    timer.stop_timer();

    double time = timer.timings.back();

    std::cout << 1000*time/(num_rep*(1.0f)) << " ms  CONV" << std::endl;
    std::cout << 1000000.0*1000*time/(num_rep*(1.0f)*apr.total_number_particles()) << " ms  million particles CONV" << std::endl;

    //check the result

    bool success = true;
    uint64_t f_c=0;
    uint64_t total_num=0;

    ExtraParticleData<float> utest_particles(apr.total_number_particles());

    apr.parameters.input_dir = options.directory;

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.apr_tree.tree_iterator();

    create_test_particles(apr,apr_iterator,tree_iterator,utest_particles,apr.particles_intensities,tree_data,stencil,stencil_size, stencil_half);

//    PixelData<uint16_t> check_mesh;
//
//    apr.interp_img(check_mesh,part_sum_dense);
//
//    std::string image_file_name = options.directory +  "check.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);
//
//    apr.interp_img(check_mesh,utest_particles);
//
//    image_file_name = options.directory +  "check_standard.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);

    //Basic serial iteration over all particles
    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {


                    if(abs(part_sum_dense[apr_iterator]-utest_particles[apr_iterator])>1){

                        //float dense = part_sum_dense[apr_iterator];

                        //float standard = utest_particles[apr_iterator];

                        //std::cout << apr_iterator.x() << " "  << apr_iterator.y() << " "  << apr_iterator.z() << " " << apr_iterator.level() << " " << dense << " " << standard << std::endl;

                        success = false;
                        f_c++;
                    }

                    total_num++;
                }
            }
        }
    }

    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << " " << f_c << " out of " << total_num <<  std::endl;
    }




}


void create_test_particles(APR<uint16_t>& apr,APRIterator& apr_iterator,APRTreeIterator& apr_tree_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<float>& part_tree,std::vector<float>& stencil, const int stencil_size, const int stencil_half){

    for (uint64_t level_local = apr_iterator.level_max(); level_local >= apr_iterator.level_min(); --level_local) {


        PixelData<float> by_level_recon;
        by_level_recon.init(apr_iterator.spatial_index_y_max(level_local),apr_iterator.spatial_index_x_max(level_local),apr_iterator.spatial_index_z_max(level_local),0);

        for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {

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

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {
            //lastly loop over particle locations and compute filter.
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    float neigh_sum = 0;
                    float counter = 0;

                    const int k = apr_iterator.y(); // offset to allow for boundary padding
                    const int i = x;

                    for (int l = -stencil_half; l < stencil_half+1; ++l) {
                        for (int q = -stencil_half; q < stencil_half+1; ++q) {
                            for (int w = -stencil_half; w < stencil_half+1; ++w) {

                                if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                    if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                        if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                            neigh_sum += stencil[counter] * by_level_recon.at(k + w, i + q, z+l);
                                        }
                                    }
                                }


                                counter++;
                            }
                        }
                    }

                    test_particles[apr_iterator] = std::roundf(neigh_sum/(1.0f*pow((float)stencil_size, 3.0f)));

                }
            }
        }




//        std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
//        TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

    }

}



bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        std::cerr << usage << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-numrep"))
    {
        result.num_rep = std::stoi(std::string(get_command_option(argv, argv + argc, "-numrep")));
    }

    if(command_option_exists(argv, argv + argc, "-stencil_size"))
    {
        result.stencil_size = std::stoi(std::string(get_command_option(argv, argv + argc, "-stencil_size")));
    }

    return result;

}
