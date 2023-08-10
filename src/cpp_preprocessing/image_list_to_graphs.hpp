#ifndef INCLUDED_IMAGE_LIST_TO_GRAPHS
#define INCLUDED_IMAGE_LIST_TO_GRAPHS

#include "create_image_segmentation.hpp"
#include "data_classes.hpp"

#include <filesystem>

#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "parallel_hashmap/phmap.h"

// in order to use SLIC you need to pass the relevant CMake parameter when compiling
// as well as install opencv_contrib
#ifdef _INCLUDE_SLIC
#include <opencv2/ximgproc/slic.hpp>
#endif


// convert each image into an image list into the graph representation and store it in OutputData
void image_list_to_graphs(std::vector<std::filesystem::path> filename_list, ParamData const paramData,
                        OutputData &outputData)
{

    // these are cached allocations for the Felzenszwalb algorithm, for efficiency
    std::vector<float> all_data;
    std::vector<float> centroid_data;

    std::vector<int> edge_conversion;
    std::vector<int> count_arr;

    std::vector<edge_int> felzenszwalb_edges;
    std::vector<int> felzenszwalb_count_arr;
    std::vector<edge_float> felzenszwalb_new_edges;
    std::vector<float> felzenszwalb_superpixel_thresholds;

    DisjointSetForest felzenszwalb_forest;

    for (auto const &filename : filename_list)
    {

        // initialise the values of the running computation for the statistics
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

        cv::Mat3b input = cv::imread(filename.c_str(), cv::IMREAD_COLOR);

        int height = input.rows;
        int width = input.cols;
        float half_height = (height - 1) / 2.0f;
        float half_width = (width - 1) / 2.0f;
        // used for normalisation
        float dim_norm = std::sqrt(static_cast<double>(height * height + width * width));
        // initialise hyperparameter variables and perform any necessary scaling
        int nr_superpixels;
        float scale = paramData.scale;
        int min_size;
        if (paramData.scale_params_to_img)
        {
            if (not paramData.scale_only_min_size)
            {
                scale *= dim_norm;
            }
            min_size = static_cast<int>(paramData.min_size * dim_norm);
        }
        else
        {
            min_size = static_cast<int>(paramData.min_size);
        }

        cv::Mat1i superpixelMap(height, width);
        phmap::flat_hash_map<int, int> mapNodeIdx;

        cv::Mat3b img(height, width);

        // apply Gaussian Blur to the image if needed
        if (paramData.sigma < 0)
        {
            img = input.clone();
        }
        else
        {
            cv::GaussianBlur(input, img, cv::Size2i(5, 5), paramData.sigma, paramData.sigma, cv::BORDER_REFLECT);
        }

#ifdef _INCLUDE_SLIC
        // perform SLIC segmentation
        if (paramData.use_slic)
        {
            cv::Ptr<cv::ximgproc::SuperpixelSLIC> slicObj = cv::ximgproc::createSuperpixelSLIC(
                img, cv::ximgproc::SLIC, paramData.region_size, paramData.compactness);
            cv::Mat1i slicSegmentation(height, width);
            slicObj->iterate();
            slicObj->getLabels(slicSegmentation);
            // assign the appropriate superpixel to each pixel
            // the superpixel indices will always be in order
            for (int curr_idx = 1, y = 0; y != height; ++y)
            {
                for (int x = 0; x != width; ++x)
                {
                    int comp = slicSegmentation.at<int>(y, x) + 1;

                    auto const found = mapNodeIdx.find(comp);
                    if (found != mapNodeIdx.end())
                        superpixelMap.at<int>(y, x) = found->second;
                    else
                    {
                        mapNodeIdx.emplace(comp, curr_idx);
                        superpixelMap.at<int>(y, x) = curr_idx;
                        ++curr_idx;
                    }
                }
            }
            nr_superpixels = mapNodeIdx.size();
        }

#endif
        // perform Felzenszwalb segmentation
        if (not paramData.use_slic)
        {
            create_image_segmentation(img, scale, min_size, nr_superpixels, felzenszwalb_edges, felzenszwalb_count_arr,
                                      felzenszwalb_new_edges, felzenszwalb_superpixel_thresholds, felzenszwalb_forest);
            // assign the appropriate superpixel to each pixel
            // the superpixel indices will always be in order
            for (int curr_idx = 1, y = 0; y != height; ++y)
            {
                for (int x = 0; x != width; ++x)
                {
                    int comp = felzenszwalb_forest.find(y * width + x) + 1;
                    auto const found = mapNodeIdx.find(comp);
                    if (found != mapNodeIdx.end())
                        superpixelMap.at<int>(y, x) = found->second;
                    else
                    {
                        mapNodeIdx.emplace(comp, curr_idx);
                        superpixelMap.at<int>(y, x) = curr_idx;
                        ++curr_idx;
                    }
                }
            }
        }

        // get the superpixel contours and associate them with the correct superpixel
        // the contours will always be inside the superpixel so if assign the contour
        // to the superpixel corresponding to one of its pixels
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(superpixelMap, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        phmap::flat_hash_map<int, int> shapeToSp;
        for (int i = 0, contours_size = contours.size(); i != contours_size; ++i)
        {
            // all points in a contour correspond to the same superpixel so just use the first
            int superpixel = superpixelMap.at<int>(contours[i][0]);
            auto found = shapeToSp.find(superpixel);
            if (found != shapeToSp.end() and i != found->second)
            {
                // with contours with holes (where the inside and outside contour are different)
                // associate the superpixel with the larger contour (the outside one)
                // this means we ignore contours of holes (of course, they will have their
                // own contour since they are a whole other superpixel)
                if (contours[i].size() > contours[found->second].size())
                {
                    shapeToSp[superpixel] = i;
                }
            }
            else
            {
                shapeToSp.emplace(superpixel, i);
            }
        }

        // simplify the contours
        std::vector<std::vector<cv::Point>> remaining_contours(nr_superpixels);
        int all_contours_size = 0;
        for (auto const &item : shapeToSp)
        {
            int current_superpixel_idx = item.first - 1;
            std::vector<cv::Point> const &curr_contour = contours[item.second];
            int epsilon_multiplier = paramData.scale_params_to_img ? curr_contour.size() : 1;
            cv::approxPolyDP(curr_contour, remaining_contours[current_superpixel_idx],
                             paramData.approx_epsilon * epsilon_multiplier, true);
            // if we have just one point in the contour, we triplicate it because of the batching assumptions
            if (remaining_contours[current_superpixel_idx].size() == 1)
            {
                remaining_contours[current_superpixel_idx].push_back(remaining_contours[current_superpixel_idx][0]);
                remaining_contours[current_superpixel_idx].push_back(remaining_contours[current_superpixel_idx][0]);
            }
            all_contours_size += remaining_contours[current_superpixel_idx].size();
        }

        // this should never be triggered
        if (remaining_contours.size() != nr_superpixels)
        {
            throw std::logic_error("number of remaining contours not matching number of components, preprocessing "
                                   "assumptions incorrect, exiting");
        }

        // create the edges between superpixels efficiently
        phmap::flat_hash_set<std::tuple<int, int>> edges;

        for (int y = 0, before_height = height - 1; y != before_height; ++y)
        {
            int x = width - 1;
            int orig_id = superpixelMap.at<int>(y, x);
            int down_id = superpixelMap.at<int>(y + 1, x);
            int left_down_id = superpixelMap.at<int>(y + 1, x - 1);

            if (orig_id != down_id)
            {
                if (orig_id < down_id)
                    edges.emplace(orig_id, down_id);
                else
                    edges.emplace(down_id, orig_id);
            }

            if (orig_id != left_down_id)
            {
                if (orig_id < left_down_id)
                    edges.emplace(orig_id, left_down_id);
                else
                    edges.emplace(left_down_id, orig_id);
            }
        }

        for (int x = 0, before_width = width - 1; x != before_width; ++x)
        {
            int y = height - 1;
            int orig_id = superpixelMap.at<int>(y, x);
            int right_id = superpixelMap.at<int>(y, x + 1);
            if (orig_id != right_id)
            {
                if (orig_id < right_id)
                    edges.emplace(orig_id, right_id);
                else
                    edges.emplace(right_id, orig_id);
            }
        }

        for (int y = 0, before_height = height - 1; y != before_height; ++y)
        {
            int x = 0;
            int orig_id = superpixelMap.at<int>(y, x);
            int down_id = superpixelMap.at<int>(y + 1, x);
            int right_id = superpixelMap.at<int>(y, x + 1);
            int right_down_id = superpixelMap.at<int>(y + 1, x + 1);

            if (orig_id != down_id)
            {
                if (orig_id < down_id)
                    edges.emplace(orig_id, down_id);
                else
                    edges.emplace(down_id, orig_id);
            }

            if (orig_id != right_id)
            {
                if (orig_id < right_id)
                    edges.emplace(orig_id, right_id);
                else
                    edges.emplace(right_id, orig_id);
            }

            if (orig_id != right_down_id)
            {
                if (orig_id < right_down_id)
                    edges.emplace(orig_id, right_down_id);
                else
                    edges.emplace(right_down_id, orig_id);
            }
        }

        for (int y = 0, before_height = height - 1; y != before_height; ++y)
        {
            for (int x = 1, before_width = width - 1; x != before_width; ++x)
            {
                int orig_id = superpixelMap.at<int>(y, x);
                int down_id = superpixelMap.at<int>(y + 1, x);
                int right_id = superpixelMap.at<int>(y, x + 1);
                int right_down_id = superpixelMap.at<int>(y + 1, x + 1);
                int left_down_id = superpixelMap.at<int>(y + 1, x - 1);

                if (orig_id != down_id)
                {
                    if (orig_id < down_id)
                        edges.emplace(orig_id, down_id);
                    else
                        edges.emplace(down_id, orig_id);
                }

                if (orig_id != right_id)
                {
                    if (orig_id < right_id)
                        edges.emplace(orig_id, right_id);
                    else
                        edges.emplace(right_id, orig_id);
                }

                if (orig_id != right_down_id)
                {
                    if (orig_id < right_down_id)
                        edges.emplace(orig_id, right_down_id);
                    else
                        edges.emplace(right_down_id, orig_id);
                }

                if (orig_id != left_down_id)
                {
                    if (orig_id < left_down_id)
                        edges.emplace(orig_id, left_down_id);
                    else
                        edges.emplace(left_down_id, orig_id);
                }
            }
        }

        int num_edges = edges.size();
        // the raw data from which all options presented in the paper and others
        // the framework supports can be obtained
        // note that edges are stored sorted twice as that is the format we need
        // for the batching and sparse tensors and it is cheaper to do it here than in the data loading
        // while not adding too much storage space
        // label 1
        // nr_superpixels 1
        // num_edges 1
        // img height 1
        // img width 1
        // all_contours_size 1
        // shape_lengths nr_superpixels
        // biggest_distance_shape_idx nr_superpixels
        // node_features 7*nr_superpixels
        // edges 4*num_edges because they are stored sorted and undirected
        // point coords with centroid at end of each shape all_contours_size*2+nr_superpixels*2
        int output_array_dim = 6 + nr_superpixels * 11 + num_edges * 4 + all_contours_size * 2;
        all_data.clear();
        all_data.resize(output_array_dim);
        centroid_data.clear();
        centroid_data.resize(2 * nr_superpixels);
        // also store the label of the image in the data according to the mapping
        if (paramData.identity_mapping)
        {
            std::string const &filenameStr = filename.filename().string();
            std::string label_name_end_pos = filenameStr.substr(0, filenameStr.find('_'));
            all_data[0] = std::stoi(label_name_end_pos);
        }
        else
        {
            std::string const &parentPath = filename.parent_path().string();
            std::string label_name = parentPath.substr(parentPath.find_last_of('/') + 1);
            all_data[0] = paramData.nameIdxMapping.find(label_name)->second;
        }

        // from now on, everything is stored based on offsets and can be reconstructed by
        // progressively reading more data (such as the number of superpixels)
        all_data[1] = nr_superpixels;
        all_data[2] = num_edges;
        all_data[3] = height;
        all_data[4] = width;
        all_data[5] = all_contours_size;
        int current_insertion_position = 6;

        // store the centroids of each superpixel
        for (auto const &item : shapeToSp)
        {
            // -1 since the findContour algorithm above takes 0 to mean the background
            // and we don't want that
            int current_superpixel_idx = item.first - 1;
            double sum_x = 0;
            double sum_y = 0;
            std::vector<cv::Point> const &curr_contour = contours[item.second];

            for (auto const &point : curr_contour)
            {
                sum_x += point.x;
                sum_y += point.y;
            }
            int curr_contour_size = curr_contour.size();
            int curr_centroid_idx_first = current_superpixel_idx * 2;
            int curr_shape_length_idx = current_insertion_position + current_superpixel_idx;
            all_data[curr_shape_length_idx] = remaining_contours[current_superpixel_idx].size();
            centroid_data[curr_centroid_idx_first] = sum_y / curr_contour_size;
            centroid_data[curr_centroid_idx_first + 1] = sum_x / curr_contour_size;
        }
        int offset_shape_max_dist_idx = current_insertion_position + nr_superpixels;
        // increase the current offset, this will always be done after a part is stored
        current_insertion_position += 2 * nr_superpixels;

        // store node data, while also computing the statistics for this image in place
        for (int y = 0; y != height; ++y)
        {
            for (int x = 0; x != width; ++x)
            {
                int curr_superpixel_idx = superpixelMap.at<int>(y, x) - 1;
                cv::Vec3b const &curr_pixel = input.at<cv::Vec3b>(y, x);
                int curr_node_position = current_insertion_position + curr_superpixel_idx * 7;
                all_data[curr_node_position] += 1;
                float pixel_r = curr_pixel[0] / 255.0f;
                float pixel_g = curr_pixel[1] / 255.0f;
                float pixel_b = curr_pixel[2] / 255.0f;
                all_data[curr_node_position + 1] += pixel_r;
                all_data[curr_node_position + 2] += pixel_g;
                all_data[curr_node_position + 3] += pixel_b;
                all_data[curr_node_position + 4] += pixel_r * pixel_r;
                all_data[curr_node_position + 5] += pixel_g * pixel_g;
                all_data[curr_node_position + 6] += pixel_b * pixel_b;
            }
        }

        float image_size = width * height;

        for (int i = 0; i != nr_superpixels; ++i)
        {
            int curr_node_position = current_insertion_position + i * 7;
            float pixel_count = all_data[curr_node_position];

            all_data[curr_node_position + 1] /= pixel_count;
            all_data[curr_node_position + 2] /= pixel_count;
            all_data[curr_node_position + 3] /= pixel_count;
            float curr_first = all_data[curr_node_position + 1];
            float curr_second = all_data[curr_node_position + 2];
            float curr_third = all_data[curr_node_position + 3];
            all_data[curr_node_position + 4] =
                std::sqrt(std::max(all_data[curr_node_position + 4] / pixel_count - curr_first * curr_first, 0.0f));
            all_data[curr_node_position + 5] =
                std::sqrt(std::max(all_data[curr_node_position + 5] / pixel_count - curr_second * curr_second, 0.0f));
            all_data[curr_node_position + 6] =
                std::sqrt(std::max(all_data[curr_node_position + 6] / pixel_count - curr_third * curr_third, 0.0f));
            all_data[curr_node_position] /= image_size;

            float curr_size = all_data[curr_node_position];

            float curr_std_first = all_data[curr_node_position + 4];
            float curr_std_second = all_data[curr_node_position + 5];
            float curr_std_third = all_data[curr_node_position + 6];

            mean_size += curr_size;
            std_size += curr_size * curr_size;

            mean_first += curr_first;
            std_first += curr_first * curr_first;

            mean_second += curr_second;
            std_second += curr_second * curr_second;

            mean_third += curr_third;
            std_third += curr_third * curr_third;

            mean_std_first += curr_std_first;
            std_std_first += curr_std_first * curr_std_first;

            mean_std_second += curr_std_second;
            std_std_second += curr_std_second * curr_std_second;

            mean_std_third += curr_std_third;
            std_std_third += curr_std_third * curr_std_third;
        }
        current_insertion_position += 7 * nr_superpixels;

        int count_sort_k = nr_superpixels * nr_superpixels + nr_superpixels;
        edge_conversion.clear();
        edge_conversion.resize(2 * num_edges);
        count_arr.clear();
        count_arr.resize(count_sort_k + 1);

        // use a counting sort for the edges, since they are small integers
        for (int curr_ins_idx = 0; auto const &item : edges)
        {
            auto [first, second] = item;
            --first;
            --second;
            edge_conversion[curr_ins_idx++] = first * nr_superpixels + second;
            edge_conversion[curr_ins_idx++] = second * nr_superpixels + first;
        }

        for (int i = 0, num_undir_edges = 2 * num_edges; i != num_undir_edges; ++i)
            ++count_arr[edge_conversion[i]];
        for (int i = 1; i <= count_sort_k; ++i)
            count_arr[i] += count_arr[i - 1];
        for (int num_undir_edges = 2 * num_edges, i = num_undir_edges - 1; i != -1; --i)
        {
            int edge_val = edge_conversion[i];
            int &count_j = count_arr[edge_val];
            --count_j;

            int insert_pos_first = current_insertion_position + count_j;
            all_data[insert_pos_first] = edge_val / nr_superpixels;
            all_data[insert_pos_first + 2 * num_edges] = edge_val % nr_superpixels;
        }

        current_insertion_position += 4 * num_edges;
        // store the contour's points
        for (int contour_idx = 0, cumulative_contour_size = 0; contour_idx != nr_superpixels; ++contour_idx)
        {
            std::vector<cv::Point> const &curr_contour = remaining_contours[contour_idx];
            int curr_centroid_pos = contour_idx * 2;
            float curr_centroid_y = centroid_data[curr_centroid_pos];
            float curr_centroid_x = centroid_data[curr_centroid_pos + 1];
            int curr_contour_size = curr_contour.size();
            // also store the node index with the biggest distance to the center
            // such that the shapes can be normalised in the batching if needed
            float max_shape_dist = -1000;
            float max_shape_angle = -1000;
            int max_contour_idx_x = -1000;
            int max_contour_idx_y = -1000;
            int max_contour_idx = -1000;
            for (int idx = 0; idx != curr_contour_size; ++idx)
            {
                cv::Point const &curr_contour_idx = curr_contour[idx];
                int next_idx = (idx + 1) % curr_contour_size;
                int current_contour_node_offset =
                    current_insertion_position + 2 * (contour_idx + cumulative_contour_size) + 2 * idx;

                float diff_x = curr_contour_idx.x - curr_centroid_x;
                float diff_y = curr_contour_idx.y - curr_centroid_y;
                float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
                float angle = std::atan2(curr_contour_idx.y - curr_centroid_y, curr_contour_idx.x - curr_centroid_x);
                all_data[current_contour_node_offset] = curr_contour_idx.y;
                all_data[current_contour_node_offset + 1] = curr_contour_idx.x;
                if (max_shape_dist == dist)
                {
                    if (max_contour_idx_y < curr_contour_idx.y)
                    {
                        max_contour_idx = idx;
                        max_shape_dist = dist;
                        max_contour_idx_x = curr_contour_idx.x;
                        max_contour_idx_y = curr_contour_idx.y;
                        max_shape_angle = angle;
                    }
                    else
                    {
                        if (max_contour_idx_y == curr_contour_idx.y)
                        {
                            if (max_contour_idx_x < curr_contour_idx.x)
                            {
                                max_contour_idx = idx;
                                max_shape_dist = dist;
                                max_contour_idx_x = curr_contour_idx.x;
                                max_contour_idx_y = curr_contour_idx.y;
                                max_shape_angle = angle;
                            }
                        }
                    }
                }
                else
                {
                    if (max_shape_dist < dist)
                    {
                        max_contour_idx = idx;
                        max_shape_dist = dist;
                        max_contour_idx_x = curr_contour_idx.x;
                        max_contour_idx_y = curr_contour_idx.y;
                        max_shape_angle = angle;
                    }
                }
            }

            mean_angle += max_shape_angle;
            std_angle += max_shape_angle * max_shape_angle;

            all_data[offset_shape_max_dist_idx + contour_idx] = max_contour_idx;
            // store the corresponding centroid as well (which is used in some framework options)
            int centroid_insertion_pos =
                current_insertion_position + 2 * (contour_idx + cumulative_contour_size + curr_contour_size);
            all_data[centroid_insertion_pos] = curr_centroid_y;
            all_data[centroid_insertion_pos + 1] = curr_centroid_x;

            cumulative_contour_size += curr_contour_size;
        }

        // we now add this image to the data objects, which requires a lock since multiple threads
        // are working at the same time
        std::lock_guard<std::mutex> output_lock(outputData.data_mutex);

        // this is done due to the batching assumptions
        if (nr_superpixels == 1 and num_edges == 0)
        {
            std::cout << "has only one connected component, not included in the dataset: " << filename << std::endl;
            continue;
        }

        outputData.offsets.push_back(outputData.data.size());

        // if we used a predefined amount of RAM stop a little before it to prevent preemtive allocation
        if (outputData.max_nr_data_points > 0 and
            (outputData.data.size() + all_data.size() + 100000 > outputData.max_nr_data_points))
        {
            throw std::logic_error{
                "The given RAM size for the output data was not sufficient, preprocessing could not be finished"
            };
        }

        // copy all the data 
        std::copy(all_data.begin(), all_data.end(), std::back_inserter(outputData.data));

        // update the dataset statistics
        outputData.total_nr_superpixels += nr_superpixels;
        outputData.mean_size += mean_size;
        outputData.std_size += std_size;

        outputData.mean_first += mean_first;
        outputData.std_first += std_first;

        outputData.mean_second += mean_second;
        outputData.std_second += std_second;

        outputData.mean_third += mean_third;
        outputData.std_third += std_third;

        outputData.mean_std_first += mean_std_first;
        outputData.std_std_first += std_std_first;

        outputData.mean_std_second += mean_std_second;
        outputData.std_std_second += std_std_second;

        outputData.mean_std_third += mean_std_third;
        outputData.std_std_third += std_std_third;

        outputData.mean_angle += mean_angle;
        outputData.std_angle += std_angle;
    }
}

#endif