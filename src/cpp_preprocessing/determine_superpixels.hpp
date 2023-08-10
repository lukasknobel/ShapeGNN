#ifndef INCLUDED_DETERMINE_SUPERPIXELS
#define INCLUDED_DETERMINE_SUPERPIXELS

#include "DisjointSetForest.hpp"
#include "data_classes.hpp"


#include <algorithm>
#include <cmath>

void determine_superpixels(int nr_vertices, int nr_edges, std::vector<edge_float> &edges, float scale,
                           std::vector<float> &superpixel_thresholds, DisjointSetForest &forest)
{

    // reinitialise forest and thresholds
    forest.create_new(nr_vertices);

    superpixel_thresholds.clear();
    superpixel_thresholds.resize(nr_vertices, scale);

    // merge superpixels according to the threshold (scale) in increasing
    // order as determined by their edge weights
    for (int i = 0; i < nr_edges; i++)
    {
        edge_float &curr_edge = edges[i];

        int first = forest.find(curr_edge.first);
        int second = forest.find(curr_edge.second);
        if (first != second)
        {
            if ((curr_edge.weight <= superpixel_thresholds[first]) &&
                (curr_edge.weight <= superpixel_thresholds[second]))
            {
                int parent = forest.merge(first, second);
                superpixel_thresholds[parent] = curr_edge.weight + scale / forest.superpixels[parent].size;
            }
        }
    }
}

#endif
