//
// Created by cheesema on 05.07.18.
//



//
// Created by cheesema on 21.01.18.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
const char* usage = R"(
Example using the APR Tree

Usage:

(using *_apr.h5 output of Example_get_apr)

Example_random_accesss -i input_apr_hdf5 -d input_directory

Note: There is no output, this file is best utilized by looking at the source code for example (test/Examples/Example_random_access.cpp) of how to code different
random access strategies on the APR.

)";


#include <algorithm>
#include <iostream>

#include "Example_apr_tree.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR <uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    std::string name = options.input;
    //remove the file extension
    name.erase(name.end() - 3, name.end());

    apr.apr_tree.init(apr);

    apr.apr_tree.fill_tree_mean_downsample(apr.particles_intensities);

    //must come after initialized.
    APRTreeIterator apr_tree_iterator = apr.apr_tree.tree_iterator();

    APRIterator apr_iterator = apr.iterator();

    timer.start_timer("APR interior tree loop");


    unsigned int delta_max = 1;
    unsigned int current_max = std::max(apr.level_max() - delta_max,apr.level_min());


    unsigned int number_parts = apr.apr_access.global_index_by_level_and_zx_end[current_max].back();

    unsigned int tree_start;
    unsigned int tree_end ;

    if(delta_max == 0){
        tree_start = 0;
        tree_end = 0;
    } else {
        tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[current_max-1].back();
        tree_end =  apr.apr_tree.tree_access.global_index_by_level_and_zx_end[current_max].back();
    }

    unsigned int number_graduated_parts = tree_end-tree_start;



    int64_t offset = number_parts - tree_start;

    std::cout << "number_parts: " << number_parts << std::endl;
    std::cout << "number_graduated_parts: " << number_graduated_parts << std::endl;
    std::cout << "combined: " << number_parts + number_graduated_parts << std::endl;

    std::vector<uint64_t> partsCombined;

    std::vector<uint64_t> partsCombined_allocate;
    partsCombined_allocate.resize(number_parts + number_graduated_parts,0);

    //Main APR particles up to level max loop

    for (unsigned int level = apr_iterator.level_min(); level <= current_max; ++level) {
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
        for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    partsCombined.push_back(apr_iterator.global_index()); //store the globalindex for checking


                    //particles up to this level are still ordered in the same way
                    partsCombined_allocate[apr_iterator] +=apr_iterator.global_index();

                }
            }
        }
    }

    std::cout << "Size parts only: " << partsCombined.size() << std::endl;

    //convolve of the interior particles that have graduated at current_max
    if(current_max < apr.level_max()) {
        unsigned int level = current_max;
        int z = 0;
        int x = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_tree_iterator)
#endif
        for (z = 0; z < apr_tree_iterator.spatial_index_z_max(level); z++) {
            for (x = 0; x < apr_tree_iterator.spatial_index_x_max(level); ++x) {
                for (apr_tree_iterator.set_new_lzx(level, z, x);
                     apr_tree_iterator.global_index() < apr_tree_iterator.end_index;
                     apr_tree_iterator.set_iterator_to_particle_next_particle()) {

                    partsCombined.push_back(apr_tree_iterator.global_index());

                    partsCombined_allocate[apr_tree_iterator.global_index() + offset] +=apr_tree_iterator.global_index();

                }
            }
        }
    }

    std::cout << "Total number of particles: " << apr.total_number_particles() << std::endl;
    std::cout << "Total number of tree particles " << apr.apr_tree.total_number_parent_cells() << std::endl;
    std::cout << "Size: " << partsCombined.size() << std::endl;

    //check the result is the same

    bool correct = true;

    for (int i = 0; i < partsCombined.size(); ++i) {
        if(partsCombined[i] == partsCombined_allocate[i]){
        } else {
            correct = false;
        }
    }

    if(!correct){
        std::cout << "FAIL" << std::endl; //this will fail when delta = 0, due to iteration order not being monotonic in index
    } else {
        std::cout << "SUCCESS" << std::endl;
    }

    //Determining the level from the input size..

    unsigned find_level = apr.level_max();
    uint64_t parts_guess = apr.total_number_particles();

    uint64_t unknown_size = partsCombined_allocate.size();



    while((unknown_size != parts_guess) && (find_level > apr.level_min()) ) {
        find_level--;

        unsigned int number_parts = apr.apr_access.global_index_by_level_and_zx_end[find_level].back();

        unsigned int tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[find_level - 1].back();
        unsigned int tree_end = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[find_level].back();

        unsigned int number_graduated_parts = tree_end - tree_start;

        parts_guess = number_parts + number_graduated_parts;
    }


    int guessed_delta = apr.level_max() - find_level;

    std::cout << " guessed max level is " << find_level << " with delta: " << guessed_delta << std::endl;
    std::cout << " actual max level is " << current_max << " with delta: " << delta_max << std::endl;



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
        std::cerr << "Usage: \"Example_random_access -i input_apr_file -d directory\"" << std::endl;
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

    return result;

}
