//
// Created by cheesema on 16/03/17.
//

#ifndef PARTPLAY_APR_HPP
#define PARTPLAY_APR_HPP


#include "../../misc/APRTimer.hpp"
#include "../../algorithm/APRParameters.hpp"
#include "../../numerics/APRCompress.hpp"
#include "../../numerics/APRReconstruction.hpp"
#include "../../io/APRWriter.hpp"
#include "APRAccess.hpp"
#include "ExtraParticleData.hpp"

//#include "../../io/TiffUtils.hpp"


template<typename ImageType>
class APR {

    template<typename> friend class APRConverter;
    template<typename> friend class APRIterator;
    friend class APRWriter;
    friend class PullingScheme;

    APRWriter apr_writer;
    APRReconstruction apr_recon;

public:
    APRAccess apr_access;

    ExtraParticleData<ImageType> particles_intensities;
    std::string name;
    APRParameters parameters;

    unsigned int orginal_dimensions(int dim) const { return apr_access.org_dims[dim]; }

    uint64_t level_max() const { return apr_access.level_max; }
    uint64_t level_min() const { return apr_access.level_min; }

    inline uint64_t spatial_index_x_max(const unsigned int level) const { return apr_access.x_num[level]; }
    inline uint64_t spatial_index_y_max(const unsigned int level) const { return apr_access.y_num[level]; }
    inline uint64_t spatial_index_z_max(const unsigned int level) const { return apr_access.z_num[level]; }

    inline uint64_t total_number_particles() const { return (apr_access).total_number_particles; }

    ///////////////////////////////////
    ///
    /// APR IO Methods (Calls members of the APRWriter class)
    ///
    //////////////////////////////////

    //basic IO
    void read_apr(std::string file_name){
        apr_writer.read_apr(*this,file_name);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name){
        return apr_writer.write_apr(*this, save_loc,file_name);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name,APRCompress<ImageType>& apr_compressor,unsigned int blosc_comp_type,unsigned int blosc_comp_level,unsigned int blosc_shuffle){
        return apr_writer.write_apr((*this),save_loc, file_name, apr_compressor,blosc_comp_type ,blosc_comp_level,blosc_shuffle);
    }

    FileSizeInfo write_apr(std::string save_loc,std::string file_name,unsigned int blosc_comp_type,unsigned int blosc_comp_level,unsigned int blosc_shuffle){
        APRCompress<ImageType> apr_compressor;
        apr_compressor.set_compression_type(0);
        return apr_writer.write_apr((*this),save_loc, file_name, apr_compressor,blosc_comp_type ,blosc_comp_level,blosc_shuffle);
    }

    //generate APR that can be read by paraview
    template<typename T>
    void write_apr_paraview(std::string save_loc,std::string file_name,ExtraParticleData<T>& parts){
        apr_writer.write_apr_paraview((*this), save_loc,file_name,parts);
    }

    //write out ExtraPartCellData
    template< typename S>
    void write_particles_only( std::string save_loc,std::string file_name,ExtraParticleData<S>& parts_extra){
        apr_writer.write_particles_only(save_loc, file_name, parts_extra);
    }

    //read in ExtraPartCellData
    template<typename T>
    void read_parts_only(std::string file_name,ExtraParticleData<T>& extra_parts){
        apr_writer.read_parts_only(file_name,extra_parts);
    }

    ////////////////////////
    ///
    ///  APR Reconstruction Methods (Calls APRReconstruction methods)
    ///
    //////////////////////////

    template<typename U,typename V>
    void interp_img(PixelData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image
        //

        apr_recon.interp_img((*this),img, parts);
    }

    template<typename U>
    void interp_depth_ds(PixelData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_depth_ds((*this),img);
    }

    template<typename U>
    void interp_depth(PixelData<U>& img){
        //
        //  Returns an image of the depth, this is down-sampled by one, as the Particle Cell solution reflects this
        //

        apr_recon.interp_level((*this), img);
    }

    template<typename U>
    void interp_type(PixelData<U>& img){
        //
        //  Interpolates the APR
        //

        apr_recon.interp_type((*this),img);
    }

    template<typename U,typename V>
    void interp_parts_smooth(PixelData<U>& out_image,ExtraParticleData<V>& interp_data,std::vector<float> scale_d = {2,2,2}){
        //
        //  Performs a smooth interpolation, based on the depth (level l) in each direction.
        //

        apr_recon.interp_parts_smooth((*this),out_image,interp_data,scale_d);
    }

    template<typename U,typename V>
    void get_parts_from_img(std::vector<PixelData<U>>& img_by_level,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using an image tree (img_by_level is a vector of images)

        //initialization of the iteration structures
        APRIterator<ImageType> apr_iterator(*this); //this is required for parallel access
        parts.data.resize(apr_iterator.total_number_particles());

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(apr_iterator)
#endif
        for (uint64_t particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);
            parts[apr_iterator] = img_by_level[apr_iterator.level()].at(apr_iterator.y(),apr_iterator.x(),apr_iterator.z());
        }
    }

    template<typename U,typename V>
    void get_parts_from_img(PixelData<U>& img,ExtraParticleData<V>& parts){
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles from an image using the nearest pixel (rounded up, i.e. next pixel after particles that sit on off pixel locations)
        //

        //initialization of the iteration structures
        APRIterator<ImageType> apr_iterator(*this); //this is required for parallel access
        uint64_t particle_number;
        parts.data.resize(apr_iterator.total_number_particles());


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            parts[apr_iterator] = img.at(apr_iterator.y_nearest_pixel(),apr_iterator.x_nearest_pixel(),apr_iterator.z_nearest_pixel());

        }

    }

    template<typename U,typename V>
    void get_parts_from_img_alt(PixelData<U>& img, ExtraParticleData<V>& parts){ //PixelData<U>& img_ds
        //
        //  Bevan Cheeseman 2016
        //
        //  Samples particles by taking pixel values when the particle aligns with a pixel, and an average of the
        //  surrounding pixel values when they do not align.

        //initialization of the iteration structures
        APRIterator<ImageType> apr_iterator(*this); //this is required for parallel access
        uint64_t particle_number;
        parts.data.resize(apr_iterator.total_number_particles());


        //PixelData<float> outimg;
        //outimg.init(img);

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator)
#endif
        for (particle_number = 0; particle_number < apr_iterator.total_number_particles(); ++particle_number) {
            //needed step for any parallel loop (update to the next part)
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            // if particle aligns with a pixel, simply take the pixel value
            if(apr_iterator.level() == apr_iterator.level_max()) {

                parts[apr_iterator] = img.mesh[apr_iterator.y() + img.y_num*apr_iterator.x() + img.x_num*img.y_num*apr_iterator.z()];

                //outimg.mesh[apr_iterator.y() + img.y_num*apr_iterator.x() + img.x_num*img.y_num*apr_iterator.z()] = parts[apr_iterator];
            } else {
                // otherwise take the mean of the neighbouring pixel values

                const uint16_t y = std::min((uint16_t)round(apr_iterator.y_global()), (uint16_t) (img.y_num - 1));
                const uint16_t x = std::min((uint16_t)round(apr_iterator.x_global()), (uint16_t) (img.x_num - 1));
                const uint16_t z = std::min((uint16_t)round(apr_iterator.z_global()), (uint16_t) (img.z_num - 1));

                const uint16_t ysh = std::max(y-1, 0);
                const uint16_t xsh = std::max(x-1, 0);
                const uint16_t zsh = std::max(z-1, 0);

                double tmp = img.at(y,x,z) / 8.0;   // x, y, z
                tmp += img.at(y, x, zsh) / 8.0;     // x, y, z-1
                tmp += img.at(y, xsh, z) / 8.0;     // x-1, y, z
                tmp += img.at(ysh, x, z) / 8.0;     // x, y-1, z
                tmp += img.at(y, xsh, zsh) / 8.0;   // x-1, y, z-1
                tmp += img.at(ysh, x, zsh) / 8.0;   // x, y-1, z-1
                tmp += img.at(ysh, xsh, z) / 8.0;   // x-1, y-1, z
                tmp += img.at(ysh, xsh, zsh) / 8.0; // x-1, y-1, z-1

                V val = (V) tmp;

                parts[apr_iterator] = val;

                //outimg.mesh[img.x_num*img.y_num*z + img.y_num*x + y] = val;

            }
        }

        //TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/sampling_img_lol.tif", outimg);

    }

};


#endif //PARTPLAY_APR_HPP
