#ifndef INCLUDED_DATA_CLASSES
#define INCLUDED_DATA_CLASSES

#include "parallel_hashmap/phmap.h"
#include <filesystem>
#include <fstream>
#include <string>

struct edge_int
{
    int weight;
    int first, second;
};

struct edge_float
{
    float weight;
    int first, second;
};

struct superpixel
{
    int rank;
    int size;
    int parent;
};

struct ParamData
{
    std::string folder_name;
    std::filesystem::path out_graph_directory;
    std::string out_file_name;
    std::string mapping;
    float sigma;
    float scale;
    float min_size;
    float approx_epsilon;
    bool use_slic;
    int region_size;
    float compactness;
    bool scale_params_to_img;
    bool scale_only_min_size;
    bool identity_mapping;
    int max_ram_for_data;
    phmap::flat_hash_map<std::string, int> nameIdxMapping;

    ParamData(char **argv)
        : folder_name(argv[1]), out_graph_directory(std::string(argv[2])), out_file_name(argv[3]), mapping(argv[4]),
          sigma(atof(argv[5])), scale(atof(argv[6])), min_size(atof(argv[7])), approx_epsilon(atof(argv[8])),
          use_slic(static_cast<bool>(atoi(argv[9]))), region_size(atoi(argv[10])), compactness(atof(argv[11])),
          scale_params_to_img(static_cast<bool>(atoi(argv[12]))),
          scale_only_min_size(static_cast<bool>(atoi(argv[13]))), max_ram_for_data(atoi(argv[14]))
    {
        // if we have an identity mapping, the filename will also give the label
        // otherwise it will give a file that contains the mapping
        identity_mapping = mapping == "identity_mapping";
        if (not identity_mapping)
        {
            std::ifstream map_file{ mapping };
            std::string current_line;
            while (getline(map_file, current_line))
            {
                int space_pos = current_line.find(' ');
                int idx = std::stoi(current_line.substr(0, space_pos));
                nameIdxMapping.emplace(current_line.substr(space_pos + 1), idx - 1);
            }
        }
    }
};

struct OutputData
{
    std::vector<float> data;
    std::vector<int64_t> offsets;
    std::mutex data_mutex;

    long long total_nr_superpixels = 0;

    double mean_size = 0.0;
    double std_size = 0.0;

    double mean_first = 0.0;
    double std_first = 0.0;

    double mean_second = 0.0;
    double std_second = 0.0;

    double mean_third = 0.0;
    double std_third = 0.0;

    double mean_std_first = 0.0;
    double std_std_first = 0.0;

    double mean_std_second = 0.0;
    double std_std_second = 0.0;

    double mean_std_third = 0.0;
    double std_std_third = 0.0;

    double mean_theta = 0.0;
    double std_theta = 0.0;

    double mean_angle = 0.0;
    double std_angle = 0.0;

    long long max_nr_data_points = -1;

    OutputData(int nr_files, int max_ram_for_data)
    {
        if (max_ram_for_data > 0)
        {
            data.reserve(static_cast<long long>(max_ram_for_data) * 1024 * 1024 / 4);
            max_nr_data_points = data.capacity();
        }
    }
};

#endif