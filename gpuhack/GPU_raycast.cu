//
// Created by cheesema on 09.03.18.
//
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__
#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif // __JETBRAINS_IDE__

#include <algorithm>
#include <vector>
#include <array>
#include <iostream>
#include <cassert>
#include <limits>
#include <chrono>
#include <iomanip>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"
#include "data_structures/Mesh/MeshData.hpp"
#include "io/TiffUtils.hpp"
#include "misc/APRTimer.hpp"

#include "thrust/device_vector.h"
#include "thrust/tuple.h"
#include "thrust/copy.h"

#include "GPUAPRAccess.hpp"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"" << argv[0] << " -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }
    if(command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    if(command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;
}



__global__ void raycast_by_level(
        const unsigned int width,
        const unsigned int height,
        const std::uint8_t minLevel,
        const std::uint8_t maxLevel,
        const float* mvp,
        const thrust::tuple<std::size_t, std::size_t>* row_info,
        const std::size_t* _chunk_index_end,
        std::size_t total_number_chunks,
        const std::uint16_t* particle_y,
        const std::uint16_t* particle_data,
        std::uint16_t* resultImage);

__global__ void reduce(std::uint16_t* inputImage,
                       const unsigned int width,
                       const unsigned int height,
                       const unsigned int levelCount,
                       std::uint16_t* resultImage);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);

    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

    APRTimer timer;
    timer.verbose_flag = true;


    int number_reps = 400;

    apr.particles_intensities.copy_data_to_gpu();

    unsigned int imageSize = apr.orginal_dimensions(1) * apr.orginal_dimensions(2);
    unsigned int imageSizeWithLevels = imageSize * (apr.level_max() - apr.level_min() + 1);

    thrust::device_vector<std::uint16_t> image;
    thrust::device_vector<std::uint16_t> finalImage;
    thrust::host_vector<std::uint16_t> finalImageHost;

    image.resize(imageSizeWithLevels, 0);
    finalImage.resize(imageSize, 0);
    finalImageHost.resize(imageSizeWithLevels, 0);

    glm::mat4 mvp = glm::mat4(1.0f);
    float* mvpArray = 0;

    std::cout << "Allocating device memory" << std::endl;

    GPUAPRAccess gpuaprAccess(apr);

    dim3 threads_dyn(32);
    dim3 blocks_dyn((gpuaprAccess.actual_number_chunks + threads_dyn.x - 1)/threads_dyn.x);

    std::cout << "Allocating matrix storage" << std::endl;

    mvp = glm::perspective(50.0f, 1024.0f/1024.0f, 0.2f, 200.0f) * glm::lookAt(glm::vec3(0.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    cudaMalloc(&mvpArray, 16 * sizeof(float));
    cudaMemcpy(mvpArray, (void*)glm::value_ptr(mvp), 16 * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    std::cout << "Will raycast " << number_reps << " " << apr.orginal_dimensions(1) << "x" << apr.orginal_dimensions(2) << " frames with " << blocks_dyn.x << "," << blocks_dyn.y << "," << blocks_dyn.z << " blocks/" << threads_dyn.x << "," << threads_dyn.y << "," << threads_dyn.z << " threads." << std::endl;
    timer.start_timer("raycasting");

    for(unsigned int run = 0; run < number_reps; run++) {
        mvp = glm::perspective(50.0f, 1024.0f/1024.0f, 0.2f, 200.0f) * glm::lookAt(
                glm::vec3(0.5f*apr.orginal_dimensions(1), 0.5f * apr.orginal_dimensions(0), 0.5f * apr.orginal_dimensions(2)),
                glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        cudaMemcpy(mvpArray, (void*)glm::value_ptr(mvp), 16 * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        raycast_by_level<<<blocks_dyn, threads_dyn>>> (
                apr.orginal_dimensions(1), apr.orginal_dimensions(2), apr.level_min(), apr.level_max(), mvpArray,
                        gpuaprAccess.gpu_access.row_info,
                        gpuaprAccess.gpu_access._chunk_index_end,
                        gpuaprAccess.actual_number_chunks,
                        gpuaprAccess.gpu_access.y_part_coord,
                        apr.particles_intensities.gpu_pointer,
                        (uint16_t*)thrust::raw_pointer_cast(image.data()));

        cudaDeviceSynchronize();
        reduce<<<32, 32>>> (
              thrust::raw_pointer_cast(image.data()), apr.orginal_dimensions(1), apr.orginal_dimensions(2), apr.level_max()-apr.level_min(), thrust::raw_pointer_cast(finalImage.data()));
        cudaDeviceSynchronize();

//        cudaMemcpy(finalImageHost, image, 1024*1024* sizeof(uint16_t) * (apr.level_max() - apr.level_min() + 1), cudaMemcpyDeviceToHost);
    }

    thrust::copy(image.begin(), image.end(), finalImageHost.begin());

    timer.stop_timer();

    std::ofstream ofp("raycast_cuda.raw", std::ios::out | std::ios::binary);
    ofp.write(reinterpret_cast<const char*>(finalImageHost.data()), imageSizeWithLevels);
    ofp.close();

    std::ofstream ofp2("raycast_cuda-high.raw", std::ios::out | std::ios::binary);
    std::vector<uint16_t> img;
    unsigned int idx = 0;
    for(auto val: finalImageHost) {
        if(idx % 4 == 0) {
            img.push_back(val);
        }
        idx++;
    }
    ofp2.write(reinterpret_cast<const char*>(img.data()), imageSize*2);
    ofp2.close();
}


//
//  This kernel checks that every particle is only visited once in the iteration
//

__device__ inline float global_position(std::uint16_t value, const std::uint8_t maxLevel, std::uint8_t level) {
    return (value + 0.5f) * std::pow(2, maxLevel - level);
}

__device__ inline float* worldToScreen(glm::mat4 mvp, float x, float y, float z, unsigned int width, unsigned int height, float* result) {
    glm::vec4 clip = mvp * glm::vec4(x, y, z, 1.0f);
    glm::vec3 ndc = glm::vec3((clip.x/clip.w - 1.0f)/2.0f, (clip.y/clip.w-1.0f)/2.0f, clip.z/clip.w);

    result[0] = ndc.x * width;
    result[1] = -ndc.y * height;
    result[2] = ndc.z;

    return result;
}

__global__ void reduce(std::uint16_t* inputImage,
                       const unsigned int width,
                       const unsigned int height,
                       const unsigned int levelCount,
                       std::uint16_t* resultImage) {
    unsigned int x = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int y = x * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    std::uint16_t result = 0;

    if(x > width || y > height) {
        return;
    }

    // gather data from input image
    #pragma unroll
    for(unsigned int i = 0; i < levelCount; i++) {
        result = max(result, inputImage[x*levelCount + y*width*levelCount + i]);
    }

    resultImage[x + width * y] = result;
}

__device__ unsigned short atomicAddShort(unsigned short* address, unsigned short val) {
    unsigned int *base_address = (unsigned int *) ((size_t) address & ~2);
    unsigned int long_val = ((size_t) address & 2) ? ((unsigned int) val << 16) : val;
    unsigned int long_old = atomicAdd(base_address, long_val);

    return ((size_t) address & 2) ? (unsigned short) (long_old >> 16) : (unsigned short) (long_old & 0xffff);
}

__global__ void raycast_by_level(
        const unsigned int width,
        const unsigned int height,
        const std::uint8_t minLevel,
        const std::uint8_t maxLevel,
        const float* mvp,
        const thrust::tuple<std::size_t, std::size_t>* row_info,
        const std::size_t* _chunk_index_end,
        std::size_t total_number_chunks,
        const std::uint16_t* particle_y,
        const std::uint16_t* particle_data,
        std::uint16_t* resultImage) {

    float global_scale = 0.05f;
    glm::mat4 theMVP = glm::make_mat4(mvp);
    const std::uint8_t levelCount = maxLevel - minLevel + 1;


    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >=total_number_chunks){
        return; //out of bounds
    }

    //load in the begin and end row indexs
    std::size_t row_begin;
    std::size_t row_end;

    if(chunk_index==0){
        row_begin = 0;
    } else {
        row_begin = _chunk_index_end[chunk_index-1] + 1; //This chunk starts the row after the last one finished.
    }

    row_end = _chunk_index_end[chunk_index];

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t current_row_key;

    for (std::size_t current_row = row_begin; current_row <= row_end; ++current_row) {
        current_row_key = thrust::get<0>(row_info[current_row]);
        if(current_row_key&1) { //checks if there any particles in the row

            particle_global_index_end = thrust::get<1>(row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
            }

            //decode the key
            std::uint16_t x = (current_row_key & KEY_X_MASK) >> KEY_X_SHIFT;
            std::uint16_t y = 0;
            std::uint16_t z = (current_row_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
            std::uint16_t intensity = 0;
            std::uint8_t level = (current_row_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;
            std::uint8_t levelNum = level - minLevel;

            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {
                y = particle_y[particle_global_index];

                float xWorld = global_position(x, maxLevel, level);
                float yWorld = global_position(y, maxLevel, level);
                float zWorld = global_position(z, maxLevel, level);
//                xWorld = x;
//                yWorld = y;
//                zWorld = z;

                intensity = particle_data[particle_global_index];

                float ndc[3];
                worldToScreen(theMVP, xWorld, yWorld, zWorld, width, height, (float*)&ndc);

//                const float* p = (const float*)glm::value_ptr(theMVP);
//                printf("matrix: %f %f %f %f / %f %f %f %f / %f %f %f %f / %f %f %f %f\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
//                printf("x/y/z -> w/h/z'/l: %f %f %f -> %f %f %f %d (%d, %d - %d)\n", xWorld, yWorld, zWorld, ndc[0], ndc[1], ndc[2], levelNum, level, minLevel, maxLevel);
                if(floor(ndc[0]) > 0 && floor(ndc[0]) < width && ndc[1] > 0 && ndc[1] < height) {
//                    if(level == 7) {
//                        printf("x/y/z/l -> w/h/z'/l: %d %d %d %d -> %f %f %f %d (%d, %d - %d)\n", x, y, z, level,
//                               ndc[0], ndc[1], ndc[2], levelNum, level, minLevel, maxLevel);
//                    }
//                    printf("x/y/z -> w/h/z'/l: %f %f %f -> %f %f %f %d (%d, %d - %d)\n", xWorld, yWorld, zWorld, ndc[0], ndc[1], ndc[2], levelNum, level, minLevel, maxLevel);
//                    unsigned int index = levelCount * (unsigned int)floor(ndc[0])
//                                         + levelCount * width * (unsigned int)floor(ndc[1])
//                                         + levelNum;
                    unsigned int index = (unsigned int)floor(ndc[0])
                                         + width * (unsigned int)floor(ndc[1]);
//                    resultImage[index] = max(resultImage[index], intensity);
                        resultImage[index] = intensity;
//                    if(intensity > 0) {
//                        resultImage[index] = max(resultImage[index], intensity);
//                    }
//                    atomicAdd((unsigned int*)resultImage[index], (unsigned int)intensity);
//                    atomicMax((unsigned int*)resultImage + index/2, (unsigned int)intensity);
//                      atomicAddShort((unsigned short*)resultImage + index , (unsigned short)intensity);
//                        resultImage[chunk_index] = 100;
//                    printf("Setting result to %d\n", intensity);
                }
            }
        }
    }
}
