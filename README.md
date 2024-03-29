# Kitten GPU LBVH

An optimized implementation of the following paper

- Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", High Performance Graphics (2012)

and the following blog posts

- https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/
- https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/

depending on [thrust](https://thrust.github.io/) and [glm](https://github.com/g-truc/glm).

## Implementation
This is partly rewritten/referenced from https://github.com/ToruNiina/lbvh to be much more optimized, simpler to use, and, importantly, bug free. It was written as a module of KittenEngine, a simulation framework I wrote for research. 

On an RTX 3090, this implementation can build a tree of 100K objects in 200us. Refitting takes 40us. Querying takes 80us. This makes it a fast solution for when performance is needed on the GPU.

Everything can be found in lbvh.cuh and lbvh.cu. 

## Interface
The LBVH is written to be as easy to use and integrate as possible with minimal use of templates. 

* For input, it takes a list of bounding boxes.
* For output, it gives a compact list of all colliding unique object ID pairs.

As bounding boxes can often be smaller in memory than the underlying primitive, the simple input can make for a performant and flexible handling of data. 
As narrow phase collision detection is often more expensive, the compact list output can facilitate higher parallelism and performance. 
These simple inputs also eliminates the need for complex templates and functors in favor of a more flexible, performant, and friendly interface. 

## Example Code

```cpp
#include "lbvh.cuh"

const int N = 100000;
const float R = 0.001f;

\\ Generate synthetic data
vector<Kitten::Bound<3, float>> points(N);
srand(1);
for (size_t i = 0; i < N; i++) {
	Kitten::Bound<3, float> b(vec3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX));
	b.pad(R);
	points[i] = b;
}

thrust::device_vector<Kitten::Bound<3, float>> d_points(points.begin(), points.end());

// Array to store results. 
thrust::device_vector<ivec2> d_res(100 * N);

// Build BVH
LBVH bvh;
bvh.compute(thrust::raw_pointer_cast(d_points.data()), N);

// Query BVH
int numCols = bvh.query(thrust::raw_pointer_cast(d_res.data()), d_res.size());

// Print results
thrust::host_vector<ivec2> res(d_res.begin(), d_res.begin() + numCols);
printf("%d collision pairs found.\n", res.size());
for (size_t i = 0; i < res.size(); i++)
	printf("(%d %d)\n", res[i].x, res[i].y);

```

That is it. 
Compute can be called multiple times to reuse allocated memory. 
In cases where a complete rebuild isnt needed, a simple refit() can ensure correctness with updated bounding boxes. 