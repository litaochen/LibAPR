//
// Modified by joeljonsson on 17.07.18.
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

#include "numerics/APRFiltering.hpp"

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

    //ExtraParticleData<float> part_sum(apr.total_number_particles());

    std::cout << 1000000.0*1000*ds_time/(num_rep*(1.0f)*apr.total_number_particles())  << " ms million DS Tree" << std::endl;
    const int stencil_half = options.stencil_size;
    const int stencil_size = 2*stencil_half + 1;

    const int y_num_stenc = (apr.apr_access.org_dims[0] > 1) ? stencil_size : 1;
    const int x_num_stenc = (apr.apr_access.org_dims[1] > 1) ? stencil_size : 1;
    const int z_num_stenc = (apr.apr_access.org_dims[2] > 1) ? stencil_size : 1;

    std::vector<int> stencil_shape = {y_num_stenc, x_num_stenc, z_num_stenc};
    std::vector<int> stencil_halves = {(y_num_stenc-1)/2, (x_num_stenc-1)/2, (z_num_stenc-1)/2};

    float norm = pow(stencil_size, apr.apr_access.number_dimensions);

    float stencil_value = 1.0f/norm;

    std::vector<PixelData<float>> stencil_vec;
    stencil_vec.resize(2);
    //stencil_vec[0].init(stencil_shape[0], stencil_shape[1], stencil_shape[2], stencil_value);


    stencil_vec[0].init(stencil_shape[0], stencil_shape[1], stencil_shape[2], 0);//stencil_value);
    stencil_vec[0].at(1,0,0) = -1;
    stencil_vec[0].at(0,1,0) = -1;
    stencil_vec[0].at(1,2,0) = -1;
    stencil_vec[0].at(2,1,0) = -1;
    stencil_vec[0].at(1,1,0) = 5;


    stencil_vec[1].init(1,1,1, 1.0f);

    //stencil.resize(norm, stencil_value);

    ExtraParticleData<float> part_sum_dense(apr.total_number_particles());

    timer.start_timer("Dense neighbour access");
    //ExtraParticleData<float> tree_data;

    APRFiltering filter_fns;


    PixelData<float> stencil_ds;
    filter_fns.downsample_stencil_alt(stencil_vec[0], stencil_ds, 1, true);

    TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/stencil_ds_alt.tif", stencil_ds);
    TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/stencil_ds_alt_input.tif", stencil_vec[0]);


    //filter_fns.convolve_equivalent(apr, stencil_vec, apr.particles_intensities, part_sum_dense);
    filter_fns.convolve_ds_stencil(apr, stencil_vec[0], apr.particles_intensities, part_sum_dense, true);
    //filter_fns.convolve(apr, stencil_vec[0], apr.particles_intensities, part_sum_dense, 0);

    PixelData<float> recon_img;
    PixelData<uint16_t> input_img;
    apr.interp_img(recon_img, part_sum_dense);
    apr.interp_img(input_img, apr.particles_intensities);

    TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/conv_recon.tif", recon_img);
    TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/conv_input.tif", input_img);


    ExtraParticleData<float> tree_data;

    apr.apr_tree.init(apr);
    apr.apr_tree.fill_tree_mean(apr, apr.apr_tree, apr.particles_intensities, tree_data);

    bool success = true;
    uint64_t f_c=0;
    uint64_t total_num=0;

    ExtraParticleData<float> utest_particles(apr.total_number_particles());

    apr.parameters.input_dir = options.directory;

    auto apr_iterator = apr.iterator();
    auto tree_iterator = apr.apr_tree.tree_iterator();



    filter_fns.create_test_particles_ds_stencil(apr,
                                                apr_iterator,
                                                tree_iterator,
                                                utest_particles,
                                                apr.particles_intensities,
                                                tree_data,
                                                stencil_vec[0],
                                                false);

    /*
    filter_fns.create_test_particles_equiv(apr,
                                           apr_iterator,
                                           tree_iterator,
                                           utest_particles,
                                           apr.particles_intensities,
                                           tree_data,
                                           stencil_vec);
    */

    //Basic serial iteration over all particles
    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) reduction(+: f_c) reduction(+: total_num)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    if(abs(part_sum_dense[apr_iterator]-utest_particles[apr_iterator])>1){

                        float dense = part_sum_dense[apr_iterator];

                        float standard = utest_particles[apr_iterator];

                        std::cout << apr_iterator.x() << " "  << apr_iterator.y() << " "  << apr_iterator.z() << " " << apr_iterator.level() << " " << dense << " " << standard << std::endl;

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

