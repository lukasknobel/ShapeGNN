#ifndef INCLUDED_CREATE_IMAGE_SEGMENTATION
#define INCLUDED_CREATE_IMAGE_SEGMENTATION

#include "data_classes.hpp"
#include "determine_superpixels.hpp"

#include <cstdlib>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

// create a pixel edge, together with its squared weight
inline void create_pixel_edge(std::vector<edge_int> &edges, int &nr_edges, int width, int y, int x,
                              cv::Vec3b const &first, int y2, int x2, cv::Mat3b const &img)
{
    cv::Vec3b const &second = img.at<cv::Vec3b>(y2, x2);
    int r_diff = first[0] - second[0];
    int g_diff = first[1] - second[1];
    int b_diff = first[2] - second[2];
    edge_int &curr_edge = edges[nr_edges];
    curr_edge.first = y * width + x;
    curr_edge.second = y2 * width + x2;
    curr_edge.weight = r_diff * r_diff + g_diff * g_diff + b_diff * b_diff;
    ++nr_edges;
}

// create the image segmentation
void create_image_segmentation(cv::Mat3b &img, float scale, int min_size, int &nr_superpixels,
                               std::vector<edge_int> &edges, std::vector<int> &count_arr,
                               std::vector<edge_float> &new_edges, std::vector<float> &superpixel_thresholds,
                               DisjointSetForest &forest)
{
    int height = img.rows;
    int width = img.cols;

    edges.clear();
    edges.resize(width * height * 4);
    int nr_edges = 0;

    // create all edges efficiently
    for (int y = 0, before_height = height - 1; y != before_height; ++y)
    {
        int x = width - 1;
        cv::Vec3b const &first = img.at<cv::Vec3b>(y, x);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x, img);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x - 1, img);
    }

    for (int x = 0, before_width = width - 1; x != before_width; ++x)
    {
        int y = height - 1;
        cv::Vec3b const &first = img.at<cv::Vec3b>(y, x);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y, x + 1, img);
    }

    for (int y = 0, before_height = height - 1; y != before_height; ++y)
    {
        int x = 0;
        cv::Vec3b const &first = img.at<cv::Vec3b>(y, x);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x, img);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y, x + 1, img);
        create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x + 1, img);
    }

    for (int y = 0, before_height = height - 1; y != before_height; ++y)
    {
        for (int x = 1, before_width = width - 1; x != before_width; ++x)
        {
            cv::Vec3b const &first = img.at<cv::Vec3b>(y, x);
            create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x, img);
            create_pixel_edge(edges, nr_edges, width, y, x, first, y, x + 1, img);
            create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x + 1, img);
            create_pixel_edge(edges, nr_edges, width, y, x, first, y + 1, x - 1, img);
        }
    }

    // since the squared edge weights are integers, we can use a counting sort to sort them
    // which is much more efficient
    int k = 255 * 255 * 3;
    count_arr.clear();
    count_arr.resize(k + 1);
    new_edges.clear();
    new_edges.resize(nr_edges);
    for (int i = 0; i != nr_edges; ++i)
        ++count_arr[edges[i].weight];
    for (int i = 1; i <= k; ++i)
        count_arr[i] += count_arr[i - 1];
    for (int i = nr_edges - 1; i != -1; --i)
    {
        edge_int &old_edges_item = edges[i];
        int &count_j = count_arr[old_edges_item.weight];
        --count_j;
        edge_float &new_edges_item = new_edges[count_j];
        new_edges_item.first = old_edges_item.first;
        new_edges_item.second = old_edges_item.second;
        new_edges_item.weight = std::sqrt(static_cast<float>(old_edges_item.weight));
    }

    determine_superpixels(width * height, nr_edges, new_edges, scale, superpixel_thresholds, forest);

    // merge superpixels that are too small into bigger ones
    for (int i = 0; i < nr_edges; ++i)
    {
        edge_float const &curr_new_edge = new_edges[i];
        int first = forest.find(curr_new_edge.first);
        int second = forest.find(curr_new_edge.second);
        if ((first != second) &&
            ((forest.superpixels[first].size < min_size) || (forest.superpixels[second].size < min_size)))
            forest.merge(first, second);
    }
    nr_superpixels = forest.nr_superpixels;
}

#endif
