//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef LIBAPR_PYAPR_HPP
#define LIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

#include "PyPixelData.hpp"
#include "PyAPRFiltering.hpp"

namespace py = pybind11;

// -------- Utility classes to be wrapped in python ----------------------------
template <typename T>
class PyAPR {
    template<typename> friend class PyPixelData;
    APR <T> apr;

public:

    PyAPR () {}

    /**
     * Reads in the given HDF5 APR file.
     *
     * @param aAprFileName
     */
    void read_apr(const std::string &aAprFileName) {
        apr.read_apr(aAprFileName);
    }

    // TODO: add more versions of write_apr, with compression options etc?
    /**
     * Writes the APR to a HDF5 file without(?) compression.
     *
     * @param aOutputFile
     */
    void write_apr(const std::string &aOutputFile) {
        apr.write_apr("", aOutputFile);
    }

    /**
     * Returns the piecewise constant reconstruction from the APR instance as a PyPixelData object. This can be cast
     * into a numpy array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> pc_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_img(apr, reconstructedImage, apr.particles_intensities);

        /*
        // this creates a copy...
        return py::array_t<T>({reconstructedImage.x_num, reconstructedImage.y_num, reconstructedImage.z_num},
                         {sizeof(T) * reconstructedImage.y_num * reconstructedImage.x_num, sizeof(T), sizeof(T) * reconstructedImage.y_num},
                         reconstructedImage.mesh.get());
        */

        //this does not copy, and can be cast to numpy.array on python side without copy (set copy=False)
        return PyPixelData<T>(reconstructedImage);
    }

    /**
     * Returns the smooth reconstruction from the APR instance as a PyPixelData object. This can be cast into a numpy
     * array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> smooth_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_parts_smooth(apr, reconstructedImage, apr.particles_intensities);

        return PyPixelData<T>(reconstructedImage);
    }

    /**
     * Sets the parameters for the APR conversion.
     *
     * @param par pyApr.APRParameters object
     */
    void set_parameters(const py::object &par) {

        if( py::isinstance<APRParameters>(par) ) {
            apr.parameters = par.cast<APRParameters>();
        } else {
            throw std::invalid_argument("Input has to be a pyApr.APRParameters object.");
        }
    }

    /**
     * Computes the APR from the input python array.
     *
     * @param input image as python (numpy) array
     */
    void get_apr_from_array(py::array &input) {

        auto buf = input.request();


        // Some checks, may need some polishing
        if( buf.ptr == nullptr ) {
            std::cerr << "Could not pass buffer in call to apr_from_array" << std::endl;
        }

        if ( !input.writeable() ) {
            std::cerr << "Input array must be writeable" << std::endl;
        }

        if( !py::isinstance<py::array_t<T>>(input) ) {
            throw std::invalid_argument("Conflicting types. Make sure the input array is of the same type as the AprType instance.");
        }

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        //TODO: fix memory/ownership passing or just revert to copying?
        input_img.init_from_mesh(buf.shape[1], buf.shape[0], buf.shape[2], ptr); // may lead to memory issues

        apr.get_apr(input_img);
    }

    /**
     * Reads in the provided tiff file and computes its APR. Note: parameters for the APR conversion should be set
     * before by using set_parameters.
     *
     * @param aInputFile path to the tiff image file
     */
    void get_apr_from_file(const std::string &aInputFile) {
        const TiffUtils::TiffInfo aTiffFile(aInputFile);

        apr.parameters.input_dir = "";
        apr.parameters.input_image_name = aInputFile;
        apr.get_apr();
    }

    /**
     * //TODO: think of copy-free options
     *
     * @return particle intensities as numpy array
     */
    py::array get_intensities() {

        auto v = new std::vector<T>(apr.particles_intensities.data);

        auto capsule = py::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<int>*>(v); });
        return py::array(v->size(), v->data(), capsule);

        /*
        auto ptr = apr.particles_intensities.data.data();

        py::capsule free_when_done(ptr, [](void *f) {
            double *foo = reinterpret_cast<double *>(f);
            std::cerr << "Element [0] = " << foo[0] << "\n";
            std::cerr << "freeing memory @ " << f << "\n";
            delete[] foo;
        });

        return py::array({apr.particles_intensities.total_number_particles()}, {sizeof(T)}, apr.particles_intensities.data.data());
        */
    }

    /*
    //TODO: use xtensor-python numpy array container?
    py::array convolve_ds(py::array &input_features, py::array &weights, py::array bias, bool normalize) {

        py::buffer_info buf = weights.request();
        auto ptr = (float *)buf.ptr;

        std::cout << "weight shape: (" << buf.shape[0] << ", " << buf.shape[1] << ", " << buf.shape[2] << ", " << buf.shape[3] << ")" << std::endl;

        std::cout << "weights(1, 0, 2, 2): " << ptr[indexat(1, 0, 2, 1, buf)] << std::endl;

        PyAPRFiltering filter_fns;

        return filter_fns.convolve_ds_cnn(apr, input, weights, bias, normalize);

    }
     */

    py::array convolve_ds_loop(py::array &input_features, py::array &weights, py::array &bias) {

        PyAPRFiltering filter_fns;

        py::buffer_info input_buf = input_features.request();
        py::buffer_info weights_buf = weights.request();
        py::buffer_info bias_buf = bias.request();

        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;

        int out_channels = weights_buf.shape[0];
        int in_channels = weights_buf.shape[1];
        int height = weights_buf.shape[2];
        int width = weights_buf.shape[3];

        std::vector<ssize_t> outshape = {out_channels, input_buf.shape[1]};
        py::array_t<float_t> output(outshape);

        py::buffer_info output_buf = output.request(true);
        auto out_ptr = (float *)output_buf.ptr;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(int i=0; i<output_buf.size; ++i) {
            out_ptr[i] = 0.0f;
        }

        //PixelData<float> output(apr.total_number_particles()/*input_buf.shape[1]*/, weights_buf.shape[0], 1, 0); // nparticles x out_channels

        for(int out=0; out<out_channels; ++out) {

            float b = bias_ptr[out];

            for (int in = 0; in < in_channels; ++in) {

                PixelData<float> stencil(height, width, 1);
                int offset = out * in_channels * width * height + in * width * height;
                int idx = 0;

                for(int y = 0; y < height; ++y) {
                    for(int x=0; x < width; ++x) {
                        stencil.at(y, x, 0) = weights_ptr[offset + idx];

                        idx++;
                    }
                }

                filter_fns.convolve_ds_stencil_loop(apr, input_features, stencil, b, output, out, in);

            }
        }

        return output;


    }

    std::vector<py::array> convolve_ds_loop_backward(py::array &grad_output, py::array &input_features, py::array &weights, py::array &bias) {

        PyAPRFiltering filter_fns;

        py::buffer_info input_buf = input_features.request();
        py::buffer_info weights_buf = weights.request();
        py::buffer_info bias_buf = bias.request();

        auto weights_ptr = (float *) weights_buf.ptr;

        int out_channels = weights_buf.shape[0];
        int in_channels = weights_buf.shape[1];
        int height = weights_buf.shape[2];
        int width = weights_buf.shape[3];

        py::array_t<float_t> grad_weights(weights_buf.shape);
        py::array_t<float_t> grad_bias(bias_buf.shape);
        py::array_t<float_t> grad_input(input_buf.shape);

        py::buffer_info grad_weights_buf = grad_weights.request(true);
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(int i=0; i<grad_weights_buf.size; ++i) {
            grad_weights_ptr[i] = 0.0f;
        }

        py::buffer_info grad_input_buf = grad_input.request(true);
        auto grad_input_ptr = (float *) grad_input_buf.ptr;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(int i=0; i<grad_input_buf.size; ++i) {
            grad_input_ptr[i] = 0.0f;
        }

        py::buffer_info grad_bias_buf = grad_bias.request(true);
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;

        for(int i=0; i<grad_bias_buf.size; ++i) {
            grad_bias_ptr[i] = 0.0f;
        }

        //PixelData<float> output(apr.total_number_particles()/*input_buf.shape[1]*/, weights_buf.shape[0], 1, 0); // nparticles x out_channels

        for(int out=0; out<out_channels; ++out) {

            for (int in = 0; in < in_channels; ++in) {

                PixelData<float> stencil(height, width, 1);
                int offset = out * in_channels * width * height + in * width * height;
                int idx = 0;

                for(int y = 0; y < height; ++y) {
                    for(int x=0; x < width; ++x) {
                        stencil.at(y, x, 0) = weights_ptr[offset + idx];

                        idx++;
                    }
                }

                //uint64_t in_offset = in * apr.total_number_particles();

                filter_fns.convolve_ds_stencil_loop_backward(apr, input_features, stencil, grad_output, grad_input, grad_weights, grad_bias, out, in);

            }
        }

        return {grad_input, grad_weights, grad_bias};

    }

    int total_num_particles() {
        return apr.total_number_particles();
    }


    py::array max_pool(py::array &input_features) {

        PyAPRFiltering filter_fns;

        //uint64_t outsize = apr.total_number_particles() - (1-pow(2.0f, -apr.apr_access.number_dimensions)) * apr.num_particles_per_level(apr.level_max());

    }


        //inline int indexat(int out, int in, int width, int height, py::buffer_info &buf){
    //    return out*buf.shape[1] * buf.shape[2] * buf.shape[3] + in * buf.shape[2] * buf.shape[3] + width * buf.shape[3] + height;
    //}

    /**
     * compute a piecewise constant reconstruction using the provided particle intensities and return the image as a
     * PyPixelData object
     *
     * @param intensities   (numpy) array
     * @return              PyPixelData reconstruction (can be cast to numpy in python w/o copy)
     */
    PyPixelData<T> recon_newints(py::array &intensities) {

        PixelData<T> recon;
        interp_img2(apr, recon, intensities);

        TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/recon_new_intensities.tif", recon);

        return PyPixelData<T>(recon);

    }


    /**
     * Compute a piecewise constant reconstruction using the provided py::array of particle intensities
     * @tparam U
     * @tparam S
     * @param apr
     * @param img
     * @param intensities
     */
    template<typename U,typename S>
    void interp_img2(APR<S>& apr, PixelData<U>& img, py::array &intensities){
        //
        //  Bevan Cheeseman 2016
        //
        //  Takes in a APR and creates piece-wise constant image

        py::buffer_info buf = intensities.request();
        auto intptr = (float *) buf.ptr;

        auto apr_iterator = apr.iterator();

        img.init(apr.orginal_dimensions(0), apr.orginal_dimensions(1), apr.orginal_dimensions(2), 0);

        int max_dim = std::max(std::max(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]), apr.apr_access.org_dims[2]);

        int max_level = ceil(std::log2(max_dim));

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

            const float step_size = pow(2, max_level - level);


#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x); apr_iterator.global_index() < apr_iterator.end_index;
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
    }

};

// -------- Templated wrapper -------------------------------------------------
template <typename DataType>
void AddPyAPR(pybind11::module &m, const std::string &aTypeString) {
    using AprType = PyAPR<DataType>;
    std::string typeStr = "Apr" + aTypeString;
    py::class_<AprType>(m, typeStr.c_str())
            .def(py::init())
            .def("read_apr", &AprType::read_apr, "Method to read HDF5 APR files")
            .def("write_apr", &AprType::write_apr, "Writes the APR instance to a HDF5 file")
            .def("reconstruct", &AprType::pc_recon, py::return_value_policy::move, "returns the piecewise constant image reconstruction as a python array")
            .def("reconstruct_smooth", &AprType::smooth_recon, py::return_value_policy::move, "returns a smooth image reconstruction as a python array")
            .def("set_parameters", &AprType::set_parameters, "Set parameters for APR conversion")
            .def("get_apr_from_array", &AprType::get_apr_from_array, "Construct APR from input array (no copy)")
            .def("get_apr_from_file", &AprType::get_apr_from_file, "Construct APR from input .tif image")
            .def("get_intensities", &AprType::get_intensities, "return the particle intensities as buffer_info")
            .def("convolve_ds_loop", &AprType::convolve_ds_loop, "convolution with stencil downsampling")
            .def("convolve_ds_loop_backward", &AprType::convolve_ds_loop_backward, "backpropagation through convolution with stencil downsampling")
            .def("recon", &AprType::recon_newints, "recon with given intensities")
            .def("nparticles", &AprType::total_num_particles, "return number of particles");
}

#endif //LIBAPR_PYAPR_HPP
