//
// Created by cheesema on 27/01/17.
//
//  These are just some base structs and functions for making writing of benchmarking easier :D
//
//  Bevan Cheeseman 2017
//
//


#ifndef PARTPLAY_BENCHMARK_HELPERS_HPP
#define PARTPLAY_BENCHMARK_HELPERS_HPP

#include "AnalysisData.hpp"
#include "TimeModel.hpp"
#include "MeshDataAF.h"
#include "SynImageClasses.hpp"
#include "GenerateTemplates.hpp"
#include "SynImagePar.hpp"
#include <stdio.h>
#include <dirent.h>


struct cmdLineOptionsBench{
    std::string template_dir = "";
    std::string template_name = "";
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    std::string description = "";
    bool template_file = false;

    bool quality_metrics_gt = false;
    bool quality_metrics_input = false;
    bool information_content = false;
    bool file_size = false;
    bool segmentation_parts = false;
    bool filters_parts = false;
    bool segmentation_mesh = false;
    bool filters_mesh = false;
    bool debug = false;
    bool nonoise = false;
    bool segmentation_eval = false;
    bool filters_eval = false;
    bool quality_true_int = false;
    bool check_scale = false;

    bool comp_perfect = false;

    float lambda = 0;
    float rel_error = 0;

    float delta = 10;

    int image_size = 128;

    float num_rep = 1;

};

bool command_option_exists_bench(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option_bench(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptionsBench read_command_line_options(int argc, char **argv){

    cmdLineOptionsBench result;

    if(argc == 1) {
        std::cerr << "Usage: \"exec -td template_director -tn template_name -d description\"" << std::endl;
        exit(1);
    }


    if(command_option_exists(argv, argv + argc, "-dir"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-dir"));
    }

    if(command_option_exists_bench(argv, argv + argc, "-quality_metrics_gt"))
    {
        result.quality_metrics_gt = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-quality_metrics_input"))
    {
        result.quality_metrics_input = true;
    }


    if(command_option_exists_bench(argv, argv + argc, "-quality_true_int"))
    {
        result.quality_true_int = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-information_content"))
    {
        result.information_content = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-file_size"))
    {
        result.file_size = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-segmentation_parts"))
    {
        result.segmentation_parts = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-segmentation_mesh"))
    {
        result.segmentation_mesh = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-segmentation_eval"))
    {
        result.segmentation_eval = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-filters_eval"))
    {
        result.filters_eval = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-filters_mesh"))
    {
        result.filters_mesh = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-filters_parts"))
    {
        result.filters_parts = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-debug"))
    {
        result.debug = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-nonoise"))
    {
        result.nonoise = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-check_scale"))
    {
        result.check_scale = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-comp_perfect"))
    {
        result.comp_perfect = true;
    }

    if(command_option_exists_bench(argv, argv + argc, "-imgsize"))
    {
        result.image_size = std::stoi(std::string(get_command_option_bench(argv, argv + argc, "-imgsize")));
    }

    if(command_option_exists_bench(argv, argv + argc, "-numrep"))
    {
        result.num_rep = std::stoi(std::string(get_command_option_bench(argv, argv + argc, "-numrep")));
    }

    if(command_option_exists_bench(argv, argv + argc, "-lambda"))
    {
        result.lambda = std::stof(std::string(get_command_option_bench(argv, argv + argc, "-lambda")));
    }

    if(command_option_exists_bench(argv, argv + argc, "-rel_error"))
    {
        result.rel_error = std::stof(std::string(get_command_option_bench(argv, argv + argc, "-rel_error")));
    }

    if(command_option_exists_bench(argv, argv + argc, "-delta"))
    {
        result.delta = std::stof(std::string(get_command_option_bench(argv, argv + argc, "-delta")));
    }



    if(command_option_exists_bench(argv, argv + argc, "-td"))
    {
        result.template_dir = std::string(get_command_option_bench(argv, argv + argc, "-td"));
        result.template_file = true;
    } else {

    }

    if(command_option_exists_bench(argv, argv + argc, "-tn"))
    {
        result.template_name = std::string(get_command_option_bench(argv, argv + argc, "-tn"));
    } else {
        //default
        result.template_name  = "sphere";
    }

    if(command_option_exists_bench(argv, argv + argc, "-d"))
    {
        result.description = std::string(get_command_option_bench(argv, argv + argc, "-d"));
    } else {
        //default
        result.description  = "unnamed";
    }





    return result;

}


std::vector<std::string> listFiles(const std::string& path,const std::string& extenstion)
{
    //
    //  Bevan Cheeseman 2017, adapted from Stack overflow code
    //
    //  For a particular folder, finds files with a certain string in their name and returns as a vector of strings, I don't think this will work on Windows.
    //


    DIR* dirFile = opendir( path.c_str() );

    std::vector<std::string> file_list;

    if ( dirFile )
    {
        struct dirent* hFile;
        errno = 0;
        while (( hFile = readdir( dirFile )) != NULL )
        {
            if ( !strcmp( hFile->d_name, "."  )) continue;
            if ( !strcmp( hFile->d_name, ".." )) continue;

            // in linux hidden files all start with '.'
            //if ( gIgnoreHidden && ( hFile->d_name[0] == '.' )) continue;

            // dirFile.name is the name of the file. Do whatever string comparison
            // you want here. Something like:
            if ( strstr( hFile->d_name, extenstion.c_str() )) {
                printf(" found a .tiff file: %s", hFile->d_name);
                std::cout << std::endl;
                file_list.push_back(hFile->d_name);
            }

        }
        closedir( dirFile );
    }

    return file_list;
}


inline bool check_file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}


struct benchmark_settings{
    //benchmark settings and defaults
    int x_num = 128;
    int y_num = 128;
    int z_num = 128;

    float voxel_size = 0.1;
    float sampling_delta = 0.1;

    float image_sampling = 200;

    float dom_size_y = 0;
    float dom_size_x = 0;
    float dom_size_z = 0;

    std::string noise_type = "poisson";

    float shift = 1000;

    float linear_shift = 0;

    float desired_I = sqrt(1000)*10;
    float N_repeats = 1;
    float num_objects = 1;
    float sig = 1;

    float int_scale_min = 1;
    float int_scale_max = 10;

    float rel_error = 0.1;

    float obj_size = 4;

    float lambda = 0;

};

void set_gaussian_psf(SynImage& syn_image_loc,benchmark_settings& bs);

void set_up_benchmark_defaults(SynImage& syn_image,benchmark_settings& bs){
    /////////////////////////////////////////
    //////////////////////////////////////////
    // SET UP THE DOMAIN SIZE

    int x_num = bs.x_num;
    int y_num = bs.y_num;
    int z_num = bs.z_num;

    ///////////////////////////////////////////////////////////////////
    //
    //  sampling properties

    //voxel size
    float voxel_size = bs.voxel_size;
    float sampling_delta = bs.sampling_delta;

    syn_image.sampling_properties.voxel_real_dims[0] = voxel_size;
    syn_image.sampling_properties.voxel_real_dims[1] = voxel_size;
    syn_image.sampling_properties.voxel_real_dims[2] = voxel_size;

    //sampling rate/delta
    syn_image.sampling_properties.sampling_delta[0] = sampling_delta;
    syn_image.sampling_properties.sampling_delta[1] = sampling_delta;
    syn_image.sampling_properties.sampling_delta[2] = sampling_delta;

    //real size of domain
    float dom_size_y = y_num*sampling_delta;
    float dom_size_x = x_num*sampling_delta;
    float dom_size_z = z_num*sampling_delta;
    syn_image.real_domain.set_domain_size(0, dom_size_y, 0, dom_size_x, 0, dom_size_z);

    bs.dom_size_y = dom_size_y;
    bs.dom_size_x = dom_size_x;
    bs.dom_size_z = dom_size_z;

    ///////////////////////////////////////////////////
    //Noise properties

    syn_image.noise_properties.gauss_var = 50;
    syn_image.noise_properties.noise_type = bs.noise_type;
    //syn_image.noise_properties.noise_type = "none";

    ////////////////////////////////////////////////////
    // Global Transforms

    float shift = bs.shift;
    syn_image.global_trans.const_shift = shift;
    float background = shift;

    float max_dim = std::max(dom_size_y,std::max(dom_size_y,dom_size_z));

    float min_grad = .5*shift/max_dim; //stop it going negative
    float max_grad = 1.5*shift/max_dim;

    Genrand_uni gen_rand;

    syn_image.global_trans.grad_y = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_x = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);
    syn_image.global_trans.grad_z = bs.linear_shift*gen_rand.rand_num(-min_grad,max_grad);




    set_gaussian_psf(syn_image,bs);

}
void update_domain(SynImage& syn_image,benchmark_settings& bs){

    syn_image.sampling_properties.voxel_real_dims[0] = bs.voxel_size;
    syn_image.sampling_properties.voxel_real_dims[1] = bs.voxel_size;
    syn_image.sampling_properties.voxel_real_dims[2] = bs.voxel_size;

    //sampling rate/delta
    syn_image.sampling_properties.sampling_delta[0] = bs.sampling_delta;
    syn_image.sampling_properties.sampling_delta[1] = bs.sampling_delta;
    syn_image.sampling_properties.sampling_delta[2] = bs.sampling_delta;

    //real size of domain
    bs.dom_size_y = bs.y_num*bs.sampling_delta;
    bs.dom_size_x = bs.x_num*bs.sampling_delta;
    bs.dom_size_z = bs.z_num*bs.sampling_delta;
    syn_image.real_domain.set_domain_size(0, bs.dom_size_y, 0, bs.dom_size_x, 0, bs.dom_size_z);

}


void set_gaussian_psf(SynImage& syn_image_loc,benchmark_settings& bs){
    ///////////////////////////////////////////////////////////////////
    //PSF properties
    syn_image_loc.PSF_properties.real_sigmas[0] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[0];
    syn_image_loc.PSF_properties.real_sigmas[1] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[1];
    syn_image_loc.PSF_properties.real_sigmas[2] = bs.sig*syn_image_loc.sampling_properties.sampling_delta[2];

    syn_image_loc.PSF_properties.I0 = 1/(pow(2*3.14159265359,1.5)*syn_image_loc.PSF_properties.real_sigmas[0]*syn_image_loc.PSF_properties.real_sigmas[1]*syn_image_loc.PSF_properties.real_sigmas[2]);

    syn_image_loc.PSF_properties.I0 = 1;

    syn_image_loc.PSF_properties.cut_th = 0.0000001;

    syn_image_loc.PSF_properties.set_guassian_window_size();

    syn_image_loc.PSF_properties.type = "gauss";

}

void generate_objects(SynImage& syn_image_loc,benchmark_settings& bs){

    Genrand_uni gen_rand;

    //remove previous objects
    syn_image_loc.real_objects.resize(0);

    // loop over the objects
    for(int id = 0;id < syn_image_loc.object_templates.size();id++) {

        //loop over the different template objects

        Real_object temp_obj;

        //set the template id
        temp_obj.template_id = 0;

        for (int q = 0; q < bs.num_objects; q++) {

            // place them randomly in the image

            temp_obj.template_id = id;

            Object_template curr_obj = syn_image_loc.object_templates[temp_obj.template_id];

            float min_dom = std::min(bs.dom_size_y,std::min(bs.dom_size_x,bs.dom_size_z));

            float max_obj = std::max(curr_obj.real_size[0],std::max(curr_obj.real_size[1],curr_obj.real_size[2]));

            if(max_obj < min_dom) {
                //have them avoid the boundary, to avoid boundary effects
                temp_obj.location[0] = gen_rand.rand_num(bs.dom_size_y * .02,
                                                         .98 * bs.dom_size_y - curr_obj.real_size[0]);
                temp_obj.location[1] = gen_rand.rand_num(bs.dom_size_x * .02,
                                                         .98 * bs.dom_size_x - curr_obj.real_size[1]);
                temp_obj.location[2] = gen_rand.rand_num(bs.dom_size_z * .02,
                                                         .98 * bs.dom_size_z - curr_obj.real_size[2]);
            } else {
                temp_obj.location[0] = .5 * bs.dom_size_y - curr_obj.real_size[0]/2;
                temp_obj.location[1] = .5 * bs.dom_size_x - curr_obj.real_size[1]/2;
                temp_obj.location[2] = .5 * bs.dom_size_z - curr_obj.real_size[2]/2;
            }
            float obj_int =  bs.desired_I;

            if(bs.int_scale_min != bs.int_scale_max) {

                obj_int = gen_rand.rand_num(bs.int_scale_min, bs.int_scale_max) * bs.desired_I;

            }
           // temp_obj.int_scale = (
                 //   ((curr_obj.real_deltas[0] * curr_obj.real_deltas[1] * curr_obj.real_deltas[2]) * obj_int) /
                 //   (curr_obj.max_sample * pow(bs.voxel_size, 3)));

            temp_obj.int_scale =  obj_int;
            syn_image_loc.real_objects.push_back(temp_obj);
        }
    }


}
void move_objects(SynImage& syn_image_loc,benchmark_settings& bs,TimeModel& t_model){
    //
    //  Moves objects around.
    //
    //


    // loop over the objects
    for(int o = 0;o < syn_image_loc.real_objects.size();o++) {


        float dx = t_model.dt*t_model.move_speed[o]*sin(t_model.theta[o])*cos(t_model.phi[o]);
        float dy =  t_model.dt*t_model.move_speed[o]*sin(t_model.theta[o])*sin(t_model.phi[o]);
        float dz =  t_model.dt*t_model.move_speed[o]*cos(t_model.theta[o]);

        //loop over the different template objects
        syn_image_loc.real_objects[o].location[0] = t_model.location[o][0] + dx;
        syn_image_loc.real_objects[o].location[1] = t_model.location[o][1] + dy;
        syn_image_loc.real_objects[o].location[2] = t_model.location[o][2] + dz;

        //update location
        t_model.location[o][0] = syn_image_loc.real_objects[o].location[0];
        t_model.location[o][1] = syn_image_loc.real_objects[o].location[1];
        t_model.location[o][2] = syn_image_loc.real_objects[o].location[2];

        t_model.theta[o] += t_model.dt*t_model.gen_rand.rand_num(0.0,1.0)*t_model.direction_speed[o];
        t_model.phi[o] += t_model.dt*t_model.gen_rand.rand_num(0.0,1.0)*t_model.direction_speed[o];

    }


}


void generate_object_center(SynImage& syn_image_loc,benchmark_settings& bs){
    //
    //  Bevan Cheeseman 2017
    //
    //  Adds one object in the center
    //

    Genrand_uni gen_rand;

    //remove previous objects
    syn_image_loc.real_objects.resize(0);

    // loop over the objects
    //loop over the different template objects

    Real_object temp_obj;

    //set the template id
    temp_obj.template_id = 0;

    Object_template curr_obj = syn_image_loc.object_templates[temp_obj.template_id];

    temp_obj.location[0] = -.5*curr_obj.real_size[0] + .5*syn_image_loc.real_domain.dims[0][1];
    temp_obj.location[1] = -.5*curr_obj.real_size[1] + .5*syn_image_loc.real_domain.dims[1][1];
    temp_obj.location[2] = -.5*curr_obj.real_size[2] + .5*syn_image_loc.real_domain.dims[2][1];

    float obj_int =  bs.int_scale_min * bs.desired_I;

    //temp_obj.int_scale = (pow(bs.voxel_size, 3)* obj_int) /
                  //  (curr_obj.max_sample *(curr_obj.real_deltas[0] * curr_obj.real_deltas[1] * curr_obj.real_deltas[2]));

    temp_obj.int_scale =  obj_int;

    syn_image_loc.real_objects.push_back(temp_obj);

}


void set_up_part_rep(SynImage& syn_image_loc,Part_rep& p_rep,benchmark_settings& bs){

    std::string image_name = "benchmark_image";

    p_rep.initialize(bs.y_num,bs.x_num,bs.z_num);

    gen_parameter_pars(syn_image_loc,p_rep.pars,image_name);

    p_rep.pars.var_th = bs.desired_I;
    p_rep.pars.rel_error = bs.rel_error;
    p_rep.len_scale = p_rep.pars.dx*pow(2.0,p_rep.pl_map.k_max+1);
    p_rep.pars.noise_sigma = sqrt(bs.shift);

    p_rep.pars.lambda = bs.lambda;

    float scale_factor = syn_image_loc.scaling_factor/syn_image_loc.object_templates[0].max_sampled_int;

    p_rep.pars.var_th = scale_factor*p_rep.pars.var_th;
    p_rep.pars.var_th_max = scale_factor*p_rep.pars.var_th_max;

    if(bs.noise_type == "none") {
        p_rep.pars.interp_type = 2;
        p_rep.pars.var_th = 1;
        p_rep.pars.lambda = -1;
    }



    get_test_paths(p_rep.pars.image_path,p_rep.pars.utest_path,p_rep.pars.output_path);
}


void process_input(cmdLineOptionsBench& options,SynImage& syn_image,AnalysisData& analysis_data,benchmark_settings& bs){
    //
    //  Bevan Cheeseman 2017
    //
    //  Takes the command line input and changes the appropriate variables
    //

    analysis_data.quality_metrics_gt = options.quality_metrics_gt;
    analysis_data.quality_metrics_input = options.quality_metrics_input;
    analysis_data.file_size = options.file_size;
    analysis_data.segmentation_parts = options.segmentation_parts;
    analysis_data.segmentation_mesh = options.segmentation_mesh;
    analysis_data.filters_parts = options.filters_parts;
    analysis_data.filters_mesh = options.filters_mesh;
    analysis_data.debug = options.debug;
    analysis_data.information_content = options.information_content;
    analysis_data.segmentation_eval = options.segmentation_eval;
    analysis_data.filters_eval = options.filters_eval;
    analysis_data.check_scale = options.check_scale;

    analysis_data.comp_perfect = options.comp_perfect;

    analysis_data.quality_true_int = options.quality_true_int;

    if(options.nonoise){
        syn_image.noise_properties.noise_type = "none";
        bs.noise_type = "none";
    }

    bs.N_repeats = options.num_rep;

    bs.x_num = options.image_size;
    bs.y_num = options.image_size;
    bs.z_num = options.image_size;

    update_domain(syn_image,bs);

    if(options.rel_error > 0){
        bs.rel_error = options.rel_error;
    } else {
        std::cout << "negative rel_error" << std::endl;
        bs.rel_error = options.rel_error;
    }

    if(options.lambda > 0){
        bs.lambda = options.lambda;
    }

}

#endif //PARTPLAY_BENCHMARK_HELPERS_HPP