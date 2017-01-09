#ifndef _filter_num_h
#define _filter_num_h
//////////////////////////////////////////////////
//
//
//  Bevan Cheeseman 2016
//
//  Filter operations (seperable)
//
//
//////////////////////////////////////////////////



#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "../data_structures/Tree/PartCellStructure.hpp"
#include "../data_structures/Tree/ExtraPartCellData.hpp"
#include "filter_help/FilterOffset.hpp"
#include "filter_help/FilterLevel.hpp"

template<typename T>
void iterate_temp_vec(std::vector<T>& temp_vec,std::vector<T>& temp_vec_depth){
    //
    //  Iterates forward these arrays
    //
    //  Copying the last value in
    //
    
    //shuffle values to the left, and then place end value, copied to end. Repeat for both
    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
    
    temp_vec.back() = temp_vec[0];
    
    std::rotate(temp_vec_depth.begin(),temp_vec_depth.begin() + 1,temp_vec_depth.end());
    
    temp_vec_depth.back() = temp_vec_depth[0];
}
template<typename T>
void iterate_temp_vec(std::vector<T>& temp_vec){
    //
    //  Iterates forward these arrays
    //
    //  Copying the last value in
    //
    
    //shuffle values to the left, and then place end value, copied to end. Repeat for both
    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
    
    temp_vec.back() = temp_vec[0];
    
}

template<typename S>
void get_neigh_check(PartCellStructure<S,uint64_t>& pc_struct){
    //
    //  Calculate Neighbours Using Iterators
    //
    //
    ParticleDataNew<float, uint64_t> part_new;
    
    part_new.initialize_from_structure(pc_struct.pc_data);
    part_new.transfer_intensities(pc_struct.part_data);
    
    component_label.initialize_structure_parts(pc_struct.part_data.particle_data);
    
    Part_timer timer;
    
    //initialize variables required
    uint64_t node_val_pc; // node variable encoding neighbour and cell information
    uint64_t node_val_part; // node variable encoding part offset status information
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    uint64_t curr_key = 0; // key used for accessing and particles and cells
    PartCellNeigh<uint64_t> neigh_part_keys; // data structure for holding particle or cell neighbours
    PartCellNeigh<uint64_t> neigh_cell_keys;
    //
    // Extra variables required
    //
    
    uint64_t status=0;
    uint64_t part_offset=0;
    uint64_t p;
    
    std::vector<int> labels;
    labels.resize(1,0);
    
    std::vector<int> neigh_labels;
    neigh_labels.reserve(10);
    
    float neigh;
    
    timer.verbose_flag = false;
    
    timer.start_timer("iterate parts");
    
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (pc_struct.pc_data.depth_min);depth <= pc_struct.pc_data.depth_max;depth++){
            //loop over the resolutions of the structure
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
            
            CurrentLevel<float,uint64_t> curr_level;
            curr_level.set_new_depth(depth,part_new);
            
            NeighOffset<float,uint64_t> neigh_y;
            neigh_y.set_new_depth(depth,part_new);
            neigh_y.set_offsets(0,0,1,0);
            
            NeighOffset<float,uint64_t> neigh_z;
            neigh_z.set_new_depth(depth,part_new);
            neigh_z.set_offsets(0,1,0,0);
            
            NeighOffset<float,uint64_t> neigh_x;
            neigh_x.set_new_depth(depth,part_new);
            neigh_x.set_offsets(1,0,0,0);
            
#pragma omp parallel for default(shared) private(p,z_,x_,j_,neigh) firstprivate(curr_level,neigh_x,neigh_z,neigh_y) if(z_num_*x_num_ > 100)
            for(z_ = 0;z_ < z_num_;z_++){
                //both z and x are explicitly accessed in the structure
                
                for(x_ = 0;x_ < x_num_;x_++){
                    
                    for(uint64_t l = 0;l < 5;l++){
                        
                        curr_level.set_new_xz(x_,z_,l,part_new);
                        neigh_x.reset_j(curr_level,part_new);
                        neigh_z.reset_j(curr_level,part_new);
                        neigh_y.reset_j(curr_level,part_new);
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num;j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,part_new);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(part_new);
                                
                                neigh_x.incriment_y_same_depth(curr_level,part_new);
                                neigh_z.incriment_y_same_depth(curr_level,part_new);
                                neigh_y.incriment_y_same_depth(curr_level,part_new);
                                
                                if(depth < pc_struct.pc_data.depth_max){
                                    
                                    neigh_x.incriment_y_child_depth(curr_level,part_new);
                                    neigh_z.incriment_y_child_depth(curr_level,part_new);
                                    neigh_y.incriment_y_child_depth(curr_level,part_new);
                                }
                                
                                if(curr_level.status==SEED){
                                    //seed loop (first particle)
                                    
                                    if(l > 0){
                                        
                                        
                                        neigh =  neigh_x.get_part(part_new.particle_data);
                                        neigh +=  neigh_z.get_part(part_new.particle_data);
                                        neigh +=  neigh_y.get_part(part_new.particle_data);
                                        //curr_level.get_part(part_new) = neigh;
                                        
                                        curr_level.iterate_y_seed();
                                        //second particle
                                        neigh_x.incriment_y_part_same_depth(curr_level,part_new);
                                        neigh_z.incriment_y_part_same_depth(curr_level,part_new);
                                        neigh_y.incriment_y_part_same_depth(curr_level,part_new);
                                        
                                        neigh = neigh_x.get_part(part_new.particle_data);
                                        neigh += neigh_z.get_part(part_new.particle_data);
                                        neigh += neigh_y.get_part(part_new.particle_data);
                                        //curr_level.get_part(part_new) = neigh;
                                        (void) neigh;
                                        
                                    } else {
                                        
                                        
                                        neigh_x.incriment_y_parent_depth(curr_level,part_new);
                                        neigh_z.incriment_y_parent_depth(curr_level,part_new);
                                        neigh_y.incriment_y_parent_depth(curr_level,part_new);
                                    }
                                    
                                } else {
                                    //non seed loop
                                    if( l == 0){
                                        
                                        neigh_x.incriment_y_parent_depth(curr_level,part_new);
                                        neigh_z.incriment_y_parent_depth(curr_level,part_new);
                                        neigh_y.incriment_y_parent_depth(curr_level,part_new);
                                        
                                        
                                        neigh = neigh_x.get_part(part_new.particle_data);
                                        neigh += neigh_z.get_part(part_new.particle_data);
                                        neigh += neigh_y.get_part(part_new.particle_data);
                                        
                                        
                                        (void) neigh;
                                        
                                        //curr_level.get_part(part_new) = neigh;
                                    }
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                curr_level.update_gap();
                                
                            }
                            
                            
                        }
                        
                    }
                }
            }
        }
        
    }
    
    timer.stop_timer();
    
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Neigh Regime New took: " << time << std::endl;
    
    
}



template<typename U,typename V>
void convolution_filter_y(PartCellStructure<U,uint64_t>& pc_struct,ExtraPartCellData<V>& filter_output){
    //
    //
    //  Loops per particle (Layer, Status vs. Non)
    //
    //
    
    std::vector<float> filter;
    
    int filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    ///////////////
    //
    //
    //
    //////////////
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    
    ////////////////////////
    //
    //  Seed loop (max resolution) example
    //
    /////////////////////////
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    timer.start_timer("y filter loop");
    
    //doing seed level (four different particle paths)
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (pc_struct.depth_max); depth >= (pc_struct.depth_min); depth--){
            
            //initialize layer iterators
            FilterLevel<uint64_t,float> curr_level;
            curr_level.set_new_depth(depth,pc_struct);
            
            curr_level.initialize_temp_vecs(filter,pc_struct);
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
            
            bool layer_active = INACTIVE;
            if(depth > (pc_struct.depth_min)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus(LOWER_RESOLUTION,layer_active);
            
            layer_plus.set_offsets(0,0,filter_offset,-1); //one layer below
            layer_plus.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth > (pc_struct.depth_min+1)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus_2(LOWER_RESOLUTION,layer_active);
            layer_plus_2.set_offsets(0,0,filter_offset,-2); //one layer below
            layer_plus_2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            //always set
            FilterOffset<uint64_t,float> layer_equal(SAME_RESOLUTION,ACTIVE);
            layer_equal.set_offsets(0,0,filter_offset,0); //one layer below
            layer_equal.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth < pc_struct.depth_max){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_minus0(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus1(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus2(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus3(HIGHER_RESOLUTION,layer_active);
            
            layer_minus0.set_offsets(0,0,filter_offset,1); //one layer below
            layer_minus0.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus1.set_offsets(1,0,filter_offset,1); //one layer below
            layer_minus1.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus2.set_offsets(0,1,filter_offset,1); //one layer below
            layer_minus2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus3.set_offsets(1,1,filter_offset,1); //one layer below
            layer_minus3.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if (depth == pc_struct.depth_max){
                
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                curr_level.iterate_temp_vecs();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update(pc_struct,curr_level);
                                layer_plus_2.incriment_y_and_update(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                
                                
                                curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                    
                                    
                                    
                                    //compute the filter
                                    curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                int y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max(y_init,curr_level.y_global - filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset-1);q++){
                                    
                                    curr_level.iterate_temp_vecs();
                                    layer_plus.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_plus_2.incriment_y_and_update(q,pc_struct,curr_level);
                                    
                                    
                                }
                            }
                        }
                    }
                }
            } else {
                
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2,layer_minus0,layer_minus1,layer_minus2,layer_minus3) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                        layer_minus0.set_new_xz(x_,z_,pc_struct);
                        layer_minus1.set_new_xz(x_,z_,pc_struct);
                        layer_minus2.set_new_xz(x_,z_,pc_struct);
                        layer_minus3.set_new_xz(x_,z_,pc_struct);
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                curr_level.iterate_temp_vecs();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update(pc_struct,curr_level);
                                layer_plus_2.incriment_y_and_update(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus0.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus1.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus2.incriment_y_and_update(pc_struct,curr_level);
                                layer_minus3.incriment_y_and_update(pc_struct,curr_level);
                                
                                curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update(pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update(pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update(pc_struct,curr_level);
                                    
                                    //compute the filter
                                    curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                int y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max(y_init,curr_level.y_global - filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset-1);q++){
                                    
                                    curr_level.iterate_temp_vecs();
                                    layer_plus.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_plus_2.incriment_y_and_update(q,pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update(q,pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update(q,pc_struct,curr_level);
                                }
                            }
                        }
                    }
                }
                
                
                
            }
            
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Filter Size: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
}

template<typename U,typename V>
void convolution_filter_y_new(PartCellStructure<U,uint64_t>& pc_struct,ExtraPartCellData<V>& filter_output){
    //
    //
    //  Loops per particle (Layer, Status vs. Non)
    //
    //
    
    std::vector<float> filter;
    
    int filter_offset = 1;
    filter.resize(filter_offset*2 +1,1.0/(filter_offset*2+1));
    
    ///////////////
    //
    //
    //
    //////////////
    
    int x_; // iteration variables
    int z_; // iteration variables
    uint64_t j_; // index variable
    
    ////////////////////////
    //
    //  Seed loop (max resolution) example
    //
    /////////////////////////
    
    Part_timer timer;
    
    timer.verbose_flag = false;
    timer.start_timer("y filter loop");
    
    //doing seed level (four different particle paths)
    
    float num_repeats = 10;
    
    for(int r = 0;r < num_repeats;r++){
        
        
        for(uint64_t depth = (pc_struct.depth_max); depth >= (pc_struct.depth_min); depth--){
            
            //initialize layer iterators
            FilterLevel<uint64_t,float> curr_level;
            curr_level.set_new_depth(depth,pc_struct);
            
            curr_level.initialize_temp_vecs_new(filter,pc_struct);
            
            const unsigned int x_num_ = pc_struct.pc_data.x_num[depth];
            const unsigned int z_num_ = pc_struct.pc_data.z_num[depth];
            
            bool layer_active = INACTIVE;
            if(depth > (pc_struct.depth_min)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus(LOWER_RESOLUTION,layer_active);
            
            layer_plus.set_offsets(0,0,filter_offset,-1); //one layer below
            layer_plus.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth > (pc_struct.depth_min+1)){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_plus_2(LOWER_RESOLUTION,layer_active);
            layer_plus_2.set_offsets(0,0,filter_offset,-2); //one layer below
            layer_plus_2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            //always set
            FilterOffset<uint64_t,float> layer_equal(SAME_RESOLUTION,ACTIVE);
            layer_equal.set_offsets(0,0,filter_offset,0); //one layer below
            layer_equal.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if(depth < pc_struct.depth_max){
                layer_active = ACTIVE;
            } else {
                layer_active = INACTIVE;
            }
            FilterOffset<uint64_t,float> layer_minus0(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus1(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus2(HIGHER_RESOLUTION,layer_active);
            FilterOffset<uint64_t,float> layer_minus3(HIGHER_RESOLUTION,layer_active);
            
            layer_minus0.set_offsets(0,0,filter_offset,1); //one layer below
            layer_minus0.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus1.set_offsets(1,0,filter_offset,1); //one layer below
            layer_minus1.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus2.set_offsets(0,1,filter_offset,1); //one layer below
            layer_minus2.set_new_depth(depth,pc_struct); //intialize for the depth
            
            layer_minus3.set_offsets(1,1,filter_offset,1); //one layer below
            layer_minus3.set_new_depth(depth,pc_struct); //intialize for the depth
            
            if (depth == pc_struct.depth_max){
                
//#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        //layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                if(curr_level.y_global > 80 && x_ == 30 && z_ == 17){
                                    int stop = 1;
                                }
                                
                                
                                curr_level.iterate_temp_vecs_new();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update_new(pc_struct,curr_level);
                                //layer_plus_2.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update_new(pc_struct,curr_level);
                                
                                
                                //curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs_new();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update_new(pc_struct,curr_level);
                                    
                                    
                                    
                                    //compute the filter
                                    //curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                int y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max((float)y_init,(float)(curr_level.y_global) - 2*filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset);q++){
                                    
                                    curr_level.iterate_temp_vecs_new();
                                    layer_plus.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    //layer_plus_2.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    
                                    
                                }
                            }
                        }
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                curr_level.update_cell(pc_struct);
                                
                                
                               
                                
                                
                                curr_level.compute_filter_new(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    curr_level.compute_filter_new(filter_output);
                                }
                                
                            } else {
                                curr_level.update_gap(pc_struct);
                            }
                        }
                    }
                }
            } else {
                
#pragma omp parallel for default(shared) private(z_,x_,j_) firstprivate(layer_plus,layer_equal,curr_level,layer_plus_2,layer_minus0,layer_minus1,layer_minus2,layer_minus3) if(z_num_*x_num_ > 100)
                for(z_ = 0;z_ < z_num_;z_++){
                    //both z and x are explicitly accessed in the structure
                    
                    for(x_ = 0;x_ < x_num_;x_++){
                        
                        //NEED TO FIX THESE THEY HAVE THE WRONG INPUT
                        //shift layers
                        layer_plus.set_new_xz(x_,z_,pc_struct);
                        layer_plus_2.set_new_xz(x_,z_,pc_struct);
                        layer_equal.set_new_xz(x_,z_,pc_struct);
                        layer_minus0.set_new_xz(x_,z_,pc_struct);
                        layer_minus1.set_new_xz(x_,z_,pc_struct);
                        layer_minus2.set_new_xz(x_,z_,pc_struct);
                        layer_minus3.set_new_xz(x_,z_,pc_struct);
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        
                        
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                //Indicates this is a particle cell node
                                curr_level.update_cell(pc_struct);
                                
                                curr_level.iterate_temp_vecs_new();
                                
                                //update and incriment
                                layer_plus.incriment_y_and_update_new(pc_struct,curr_level);
                                //layer_plus_2.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_equal.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_minus0.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_minus1.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_minus2.incriment_y_and_update_new(pc_struct,curr_level);
                                layer_minus3.incriment_y_and_update_new(pc_struct,curr_level);
                                
                                //curr_level.compute_filter(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    //iterate the vectors
                                    curr_level.iterate_temp_vecs_new();
                                    
                                    //update them with new values
                                    layer_equal.incriment_y_and_update_new(pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update_new(pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update_new(pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update_new(pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update_new(pc_struct,curr_level);
                                    
                                    //compute the filter
                                    //curr_level.compute_filter(filter_output);
                                }
                                
                                
                            } else {
                                // Jumps the iteration forward, this therefore also requires computation of an effective boundary condition
                                
                                int y_init = curr_level.y_global;
                                
                                curr_level.update_gap(pc_struct);
                                
                                //will need to initialize things here..
                                y_init = std::max(y_init,curr_level.y_global - 2*filter_offset);
                                
                                for(uint64_t q = y_init;q < curr_level.y_global + (filter_offset);q++){
                                    
                                    curr_level.iterate_temp_vecs_new();
                                    layer_plus.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    layer_equal.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    //layer_plus_2.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    
                                    layer_minus0.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    layer_minus1.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    layer_minus2.incriment_y_and_update_new(q,pc_struct,curr_level);
                                    layer_minus3.incriment_y_and_update_new(q,pc_struct,curr_level);
                                }
                            }
                        }
                        
                        curr_level.set_new_xz(x_,z_,pc_struct);
                        //the y direction loop however is sparse, and must be accessed accordinagly
                        for(j_ = 0;j_ < curr_level.j_num_();j_++){
                            
                            //particle cell node value, used here as it is requried for getting the particle neighbours
                            bool iscell = curr_level.new_j(j_,pc_struct);
                            
                            if (iscell){
                                curr_level.update_cell(pc_struct);
                                curr_level.compute_filter_new(filter_output);
                                
                                if(curr_level.status_()==SEED){
                                    //iterate forward
                                    curr_level.iterate_y_seed();
                                    curr_level.compute_filter_new(filter_output);
                                }
                                
                            } else {
                                curr_level.update_gap(pc_struct);
                            }
                        }
                        
                    }
                }
                
                
                
            }
            
        }
        
    }
    
    
    
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Particle Filter Size: " << pc_struct.get_number_parts() << " took: " << time << std::endl;
    
}


template<typename U>
void convolution_filter_pixels(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 1;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(filter.size());
    
    uint64_t offset_;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_) firstprivate(temp_vec)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                for(k = 0;k < y_num;k++){
                    
                    std::rotate(temp_vec.begin(),temp_vec.begin() + 1,temp_vec.end());
                    
                    offset_ = std::min(k + filter_offset,y_num);
                    
                    temp_vec.back() = input_data.mesh[j*x_num*y_num + i*y_num + k + offset_];
                    
                    for(uint64_t f = 0;f < filter.size();f++){
                        
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[f]*filter[f];
                    }
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}
template<typename U>
void convolution_filter_pixels_temp(PartCellStructure<U,uint64_t>& pc_struct,uint64_t y_num,uint64_t x_num,uint64_t z_num){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //
    
    Mesh_data<U> input_data;
    Mesh_data<U> output_data;
    input_data.initialize((int)y_num,(int)x_num,(int)z_num,23);
    output_data.initialize((int)y_num,(int)x_num,(int)z_num,0);
    
    std::vector<float> filter;
    
    Part_timer timer;
    timer.verbose_flag = false;
    timer.start_timer("full previous filter");
    
    uint64_t filter_offset = 6;
    filter.resize(filter_offset*2 +1,1);
    
    std::vector<U> temp_vec;
    temp_vec.resize(y_num,0);
    
    uint64_t offset_min;
    uint64_t offset_max;
    
    uint64_t j = 0;
    uint64_t k = 0;
    uint64_t i = 0;
    
    float num_repeats = 50;
    
    for(int r = 0;r < num_repeats;r++){
        
#pragma omp parallel for default(shared) private(j,i,k,offset_min,offset_max) firstprivate(temp_vec)
        for(j = 0; j < z_num;j++){
            for(i = 0; i < x_num;i++){
                
                //for(k = 0;k < y_num;k++){
                //  temp_vec[k] = input_data.mesh[j*x_num*y_num + i*y_num + k];
                //}
                
                for(k = 0;k < y_num;k++){
                    
                    offset_max = std::min((uint64_t)(k + filter_offset),(uint64_t)(y_num-1));
                    offset_min = std::max((uint64_t)(k - filter_offset),(uint64_t)0);
                    
                    uint64_t f = 0;
                    for(uint64_t c = offset_min;c <= offset_max;c++){
                        
                        //output_data.mesh[j*x_num*y_num + i*y_num + k] += temp_vec[c]*filter[f];
                        output_data.mesh[j*x_num*y_num + i*y_num + k] += input_data.mesh[j*x_num*y_num + i*y_num + c]*filter[f];
                        f++;
                    }
                    
                }
            }
        }
        
    }
    
    
    timer.stop_timer();
    float time = (timer.t2 - timer.t1)/num_repeats;
    
    std::cout << " Pixel Filter Size: " << (x_num*y_num*z_num) << " took: " << time << std::endl;
    
}



#endif