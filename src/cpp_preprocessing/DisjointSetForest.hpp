#ifndef INCLUDED_DISJOINT_SET_FOREST
#define INCLUDED_DISJOINT_SET_FOREST

#include <vector>

#include "data_classes.hpp"

// The disjoint-set forest data structure allows very efficient
// querying of the parent of a node (which will be the superpixel of a pixel)
struct DisjointSetForest
{

    std::vector<superpixel> superpixels;
    int nr_superpixels;

    // reinitialise the forest (used since we cache the allocation)
    void create_new(int elements)
    {
        superpixels.clear();
        superpixels.resize(elements);
        nr_superpixels = elements;
        for (int i = 0; i < elements; i++)
        {
            superpixels[i] = { .rank = 0, .size = 1, .parent = i };
        }
    }

    // merge two superpixels and return the new parent
    int merge(int x, int y)
    {
        superpixel &superpixels_x = superpixels[x];
        superpixel &superpixels_y = superpixels[y];
        int parent;
        if (superpixels_x.rank > superpixels_y.rank)
        {
            superpixels_x.size += superpixels_y.size;
            superpixels_y.parent = x;
            parent = x;
        }
        else
        {
            superpixels_y.size += superpixels_x.size;
            if (superpixels_x.rank == superpixels_y.rank)
            {
                ++superpixels_y.rank;
            }
            superpixels_x.parent = y;
            parent = y;
        }
        --nr_superpixels;
        return parent;
    }

    // find the superpixel of a pixel
    int find(int x)
    {
        int root = x;
        int root_parent = superpixels[root].parent;
        while (root_parent != root)
        {
            root = root_parent;
            root_parent = superpixels[root].parent;
        }

        int x_parent = superpixels[x].parent;
        while (x_parent != root)
        {
            superpixels[x].parent = root;
            x = x_parent;
            x_parent = superpixels[x].parent;
        }
        return root;
    }
};

#endif
