//
// Created by cheesema on 27/01/17.
//

#include "benchmark_increase_domain.hpp"


////////////////////////
//
//  Bevan Cheeseman 2017
//
//  Generating Increasing Domain Size
//
////////////////////////



int main(int argc, char **argv) {


    //////////////////////////////////////////
    //
    //
    //  First set up the synthetic problem to be solved
    //
    //
    ///////////////////////////////////////////

    cmdLineOptionsBench options = read_command_line_options(argc,argv);

    SynImage syn_image;

    std::string image_name = options.template_name;




    ///////////////////////////////////////////////////////////////////
    //  PSF properties
    //////////////////////////////////////////////////////////////////

    benchmark_settings bs;


    bs.sig = 2;

    set_up_benchmark_defaults(syn_image,bs);

    /////////////////////////////////////////////
    // GENERATE THE OBJECT TEMPLATE

    std::cout << "Generating Templates" << std::endl;

    bs.obj_size = 4;

    obj_properties obj_prop(bs.obj_size,bs.sig);


    Object_template  basic_object = get_object_template(options,obj_prop);

    syn_image.object_templates.push_back(basic_object);


    /////////////////////////////////////////////////////////////////
    //
    //
    //  Now perform the experiment looping over and generating datasets x times.
    //
    //
    //
    //////////////////////////////////////////////////////////////////

    std::cout << "BENCHMARK INCREASE IMAGE SIZE" << std::endl;

    AnalysisData analysis_data(options.description,"Benchmark fixed number of spheres with increasing sized imaging domain",argc,argv);

    analysis_data.create_float_dataset("num_objects",0);

    analysis_data.segmentation = true;
    analysis_data.filters = false;

    // In this case we are increasing the number of objects

    std::vector<int> image_size;

    float min_size = 50;
    float max_size = 600;
    float delta = 50;

    for (int i = min_size; i < max_size; i = i + delta) {
        image_size.push_back(i);
    }

    bs.num_objects = 5;

    bs.N_repeats = 1; // so you have this many realisations at the parameter set
    int N_par = (int)image_size.size(); // this many different parameter values to be run

    Part_timer b_timer;
    b_timer.verbose_flag = false;


    for (int j = 0;j < N_par;j++){

        for(int i = 0; i < bs.N_repeats; i++){

            b_timer.start_timer("one_it");

            ///////////////////////////////
            //
            //  Individual synthetic image parameters
            //
            ///////////////////////////////

            analysis_data.get_data_ref<float>("num_objects")->data.push_back(bs.num_objects);
            analysis_data.part_data_list["num_objects"].print_flag = true;

            SynImage syn_image_loc = syn_image;

            std::cout << "Par: " << j << " of " << N_par << " Rep: " << i << " of " << bs.N_repeats << std::endl;

            Mesh_data<uint16_t> input_image;


            /////////////////////////////////////////
            //////////////////////////////////////////
            // SET UP THE DOMAIN SIZE

            bs.x_num = image_size[j];
            bs.y_num = image_size[j];
            bs.z_num = image_size[j];

            update_domain(syn_image_loc,bs);

            //Generate objects

            generate_objects(syn_image_loc,bs);


            ///////////////////////////////
            //
            //  Generate the image
            //
            ////////////////////////////////

            MeshDataAF<uint16_t> gen_image;

            syn_image_loc.generate_syn_image(gen_image);

            Mesh_data<uint16_t> input_img;

            copy_mesh_data_structures(gen_image,input_img);


            ///////////////////////////////
            //
            //  Get the APR
            //
            //////////////////////////////

            Part_rep p_rep;

            set_up_part_rep(syn_image_loc,p_rep,bs);

            // Get the APR

            PartCellStructure<float,uint64_t> pc_struct;

            bench_get_apr(input_img,p_rep,pc_struct,analysis_data);

            ///////////////////////////////
            //
            //  Calculate analysis of the result
            //
            ///////////////////////////////

            produce_apr_analysis(input_img,analysis_data,pc_struct,syn_image_loc,p_rep.pars);


            af::sync();
            af::deviceGC();

            b_timer.stop_timer();

        }
    }

    //write the analysis output
    analysis_data.write_analysis_data_hdf5();

    af::info();

    return 0;


}
