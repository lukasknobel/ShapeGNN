#include "image_list_to_graphs.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>


#include <xtensor.hpp>
#include <xtensor/xnpy.hpp>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    std::string infoString = R"END(
NOTE: you do not need to call this executable yourself: call
the Python code and it will handle creating the dataset if needed.

Incorrect usage: all following parameters should be passed in order:

input_folder: the folder with the input images
output_folder: the path where to output the graph dataset
file_name: general name for the outputted dataset, should be the split name (train or test)
mapping: whether to perform an identity mapping (pass "identity_mapping" without "") or the file that gives the mapping
sigma: gaussian kernel hyperparameter for Felzenszwalb
scale: scale hyperparameter for Felzenszwalb
min_size: minimum size of a superpixel for Felzenszwalb
approx_epsilon: the epsilon used as maximum error for the shape approximation
use_slic: whether to use the SLIC segmentation
region_size: hyperparameter for SLIC
compactness: hyperparameter for SLIC
scale_params_to_img: whether to scale hyperparameters to image size for Felzenszwalb
scale_only_min_size: if the above is set to true, only scale min_size
max_ram_for_data: the amount of MB that the program will attempt to reserve for the dataset, to prevent the doubling strategy of std::vector to save memory
)END";
    if (argc != 15)
    {
        std::cout<<infoString<<std::endl;
        return 1;
    }
    ParamData paramData(argv);

    std::cout << "input_path: " << paramData.folder_name << ", output_path: " << paramData.out_graph_directory << '\n';
    if (not fs::exists(paramData.out_graph_directory))
        fs::create_directory(paramData.out_graph_directory);
    std::vector<fs::path> paths;

    // obtain all image file names and store them in a vector
    for (auto const &entry : fs::recursive_directory_iterator(paramData.folder_name))
    {

        auto entry_path = entry.path();
        if (entry_path.extension() == ".jpg" || entry_path.extension() == ".png" || entry_path.extension() == ".JPEG")
        {
            paths.push_back(entry_path);
        }
    }

    // so they can be split up in a similar amount for each thread
    std::vector<std::thread> threads;
    unsigned int nr_threads = std::thread::hardware_concurrency();
    int nr_files = paths.size();
    std::cout << "nr of threads used: " << nr_threads << ", nr of files to be processed: " << nr_files
              << ", max ram for dataset file (in MB): " << paramData.max_ram_for_data << std::endl;
    int avg_nr_files = nr_files / nr_threads;
    int begin_offset = 0;
    int end_offset = avg_nr_files;

    OutputData outputData(nr_files, paramData.max_ram_for_data);

    // create threads to process their images
    for (int i = 0; i < nr_threads - 1; ++i)
    {
        threads.emplace_back(image_list_to_graphs,
                             std::vector<fs::path>(paths.begin() + begin_offset, paths.begin() + end_offset), paramData,
                             std::ref(outputData));
        begin_offset += avg_nr_files;
        end_offset += avg_nr_files;
    }
    threads.emplace_back(image_list_to_graphs, std::vector<fs::path>(paths.begin() + begin_offset, paths.end()),
                         paramData, std::ref(outputData));

    // wait for all threads to be finished
    for (auto &thread : threads)
    {
        thread.join();
    }

    // create the numpy arrays corresponding to the dataset
    // the raw data
    xt::xarray<float> outputTensor =
        xt::adapt(outputData.data, std::vector<int64_t>{ static_cast<int64_t>(outputData.data.size()) });
    xt::dump_npy(paramData.out_graph_directory / fs::path(paramData.out_file_name).replace_extension(".npy"),
                 outputTensor);
    outputData.offsets.push_back(outputData.data.size());
    // the offsets determining where each image graph begins
    xt::xarray<int64_t> outputOffsetsTensor =
        xt::adapt(outputData.offsets, std::vector<int64_t>{ static_cast<int64_t>(outputData.offsets.size()) });
    xt::dump_npy(paramData.out_graph_directory /
                     fs::path(paramData.out_file_name + "_offsets").replace_extension(".npy"),
                 outputOffsetsTensor);
    
    // finish the dataset statistics computation and store that as well
    // this is necessary because the mean and std are done in-place
    outputData.mean_size /= outputData.total_nr_superpixels;
    outputData.std_size = std::sqrt(std::max(
        outputData.std_size / outputData.total_nr_superpixels - outputData.mean_size * outputData.mean_size, 0.0));

    outputData.mean_first /= outputData.total_nr_superpixels;
    outputData.std_first = std::sqrt(std::max(
        outputData.std_first / outputData.total_nr_superpixels - outputData.mean_first * outputData.mean_first, 0.0));

    outputData.mean_second /= outputData.total_nr_superpixels;
    outputData.std_second = std::sqrt(std::max(outputData.std_second / outputData.total_nr_superpixels -
                                                   outputData.mean_second * outputData.mean_second,
                                               0.0));

    outputData.mean_third /= outputData.total_nr_superpixels;
    outputData.std_third = std::sqrt(std::max(
        outputData.std_third / outputData.total_nr_superpixels - outputData.mean_third * outputData.mean_third, 0.0));

    outputData.mean_std_first /= outputData.total_nr_superpixels;
    outputData.std_std_first = std::sqrt(std::max(outputData.std_std_first / outputData.total_nr_superpixels -
                                                      outputData.mean_std_first * outputData.mean_std_first,
                                                  0.0));

    outputData.mean_std_second /= outputData.total_nr_superpixels;
    outputData.std_std_second = std::sqrt(std::max(outputData.std_std_second / outputData.total_nr_superpixels -
                                                       outputData.mean_std_second * outputData.mean_std_second,
                                                   0.0));

    outputData.mean_std_third /= outputData.total_nr_superpixels;
    outputData.std_std_third = std::sqrt(std::max(outputData.std_std_third / outputData.total_nr_superpixels -
                                                      outputData.mean_std_third * outputData.mean_std_third,
                                                  0.0));

    outputData.mean_angle /= outputData.total_nr_superpixels;
    outputData.std_angle = std::sqrt(std::max(
        outputData.std_angle / outputData.total_nr_superpixels - outputData.mean_angle * outputData.mean_angle, 0.0));

    xt::xarray<float> outputStatistics = {
        static_cast<float>(outputData.mean_size),       static_cast<float>(outputData.std_size),

        static_cast<float>(outputData.mean_first),      static_cast<float>(outputData.std_first),

        static_cast<float>(outputData.mean_second),     static_cast<float>(outputData.std_second),

        static_cast<float>(outputData.mean_third),      static_cast<float>(outputData.std_third),

        static_cast<float>(outputData.mean_std_first),  static_cast<float>(outputData.std_std_first),

        static_cast<float>(outputData.mean_std_second), static_cast<float>(outputData.std_std_second),

        static_cast<float>(outputData.mean_std_third),  static_cast<float>(outputData.std_std_third),

        static_cast<float>(outputData.mean_angle),      static_cast<float>(outputData.std_angle)
    };

    xt::dump_npy(paramData.out_graph_directory /
                     fs::path(paramData.out_file_name + "_statistics").replace_extension(".npy"),
                 outputStatistics);

    return 0;
}
