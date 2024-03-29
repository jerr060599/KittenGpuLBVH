#pragma once
// Jerry Hsu, 2024

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "KittenEngine/includes/modules/Common.h"
#include "KittenEngine/includes/modules/Bound.h"

namespace Kitten {
	/// <summary>
	/// Very simple high-performance GPU LBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
	/// </summary>
	class LBVH {
	public:
		typedef Bound<3, float> aabb;
		typedef glm::vec3 vec_type;

		struct node {
			uint32_t parentIdx;		// Parent node
			union {
				int leftIdx;	// Index of left child node.
				int objIdx;		// Index of the aabb in the original data array
			};
			int rightIdx;		// Index of right child node.
			int fence;			// This subtree have indices between fence and current index. Negative if leaf

			__device__ __host__ bool isLeaf() const {
				return fence < 0;
			}
		};

	private:
		thrust::device_ptr<aabb> d_objs = nullptr;
		size_t numObjs = 0;

		thrust::device_vector<int> d_flags;			// Flags used for updating the tree

		thrust::device_vector<uint32_t> d_morton;	// Morton codes for each lead
		thrust::device_vector<int> d_objIDs;		// Object ID for each leaf
		thrust::device_vector<node> d_nodes;		// The internal tree nodes
		thrust::device_vector<aabb> d_aabbs;		// AABB of each node

		// Performs a refit without updating the d_aabbs array
		void refitNoCopy();

	public:
		/// <summary>
		/// Refits an existing aabb tree once compute() has been called.
		/// Does not recompute the tree structure but only the AABBs.
		/// </summary>
		void refit();

		/// <summary>
		/// Allocates memory and builds the LBVH from a list of AABBs.
		/// Can be called multiple times for memory reuse.
		/// </summary>
		/// <param name="devicePtr">The device pointer containing the AABBs</param>
		/// <param name="size">The number of AABBs</param>
		void compute(aabb* devicePtr, size_t size);

		/// <summary>
		/// Tests this BVH against another BVH. Outputs unique collision pairs.
		/// The calling BVH should be the smaller one for best performance. 
		/// </summary>
		/// <param name="d_res">Device pointer with pairs containing (in order) the calling BVH object ID and then the other BVH object ID.</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <param name="other">The other BVH</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(ivec2* d_res, size_t resSize, LBVH* other) const;

		/// <summary>
		/// Tests this BVH against itself. Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(ivec2* d_res, size_t resSize) const;

		// Does a self check of the BVH structure for debugging purposes.
		void bvhSelfCheck() const;
	};

	// Tests the LBVH with a simple test case of 100k objects.
	void testLBVH();
}