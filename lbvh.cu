#include "lbvh.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/swap.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <unordered_set>

namespace Kitten {

#pragma region LBVHDevice

	namespace LBVHKernels {
		// Computes the morton codes for each AABB
		template<typename T>
		__global__ void mortonKernel(Bound<3, T>* aabbs, uint32_t* codes, int* ids, Bound<3, T> wholeAABB, int size) {
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;
			vec3 coord = wholeAABB.normCoord(aabbs[tid].center()) * 1024.f;
			codes[tid] = Kitten::getMorton(clamp(ivec3(coord), ivec3(0), ivec3(1023)));
			ids[tid] = tid;
		}

		// Uses CUDA intrinsics for counting leading zeros
		__device__ inline int commonUpperBits(const uint64_t lhs, const uint64_t rhs) {
			return ::__clzll(lhs ^ rhs);
		}

		// Merges morton code with its index to output a sorted unique 64-bit key.
		__device__ inline uint64_t mergeIdx(const uint32_t code, const int idx) {
			return ((uint64_t)code << 32ul) | (uint64_t)idx;
		}

		__device__ inline ivec2 determineRange(uint32_t const* mortonCodes,
			const uint32_t numObjs, uint32_t idx) {

			// This is the root node
			if (idx == 0)
				return ivec2(0, numObjs - 1);

			// Determine direction of the range
			const uint64_t selfCode = mergeIdx(mortonCodes[idx], idx);
			const int lDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx - 1], idx - 1));
			const int rDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx + 1], idx + 1));
			const int d = (rDelta > lDelta) ? 1 : -1;

			// Compute upper bound for the length of the range
			const int minDelta = thrust::min(lDelta, rDelta);
			int lMax = 2;
			int i;
			while ((i = idx + d * lMax) >= 0 && i < numObjs) {
				if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) <= minDelta) break;
				lMax <<= 1;
			}

			// Find the exact range by binary search
			int t = lMax >> 1;
			int l = 0;
			while (t > 0) {
				i = idx + (l + t) * d;
				if (0 <= i && i < numObjs)
					if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) > minDelta)
						l += t;
				t >>= 1;
			}

			unsigned int jdx = idx + l * d;
			if (d < 0) thrust::swap(idx, jdx); // Make sure that idx < jdx
			return ivec2(idx, jdx);
		}

		__device__ inline uint32_t findSplit(uint32_t const* mortonCodes,
			const uint32_t first, const uint32_t last) {

			const uint64_t firstCode = mergeIdx(mortonCodes[first], first);
			const uint64_t lastCode = mergeIdx(mortonCodes[last], last);
			const int deltaNode = commonUpperBits(firstCode, lastCode);

			// Binary search for split position
			int split = first;
			int stride = last - first;
			do {
				stride = (stride + 1) >> 1;
				const int middle = split + stride;
				if (middle < last)
					if (commonUpperBits(firstCode, mergeIdx(mortonCodes[middle], middle)) > deltaNode)
						split = middle;
			} while (stride > 1);

			return split;
		}

		// Builds out the leaf nodes and reorders the AABBs to match the leaf ordering
		__global__ void lbvhBuildLeafKernel(LBVH::node* leafNodes,
			Bound<3, float>* leafAABBs, Bound<3, float>* objAABBs,
			int const* objIDs, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			LBVH::node node;
			node.objIdx = objIDs[tid];
			leafAABBs[tid] = objAABBs[node.objIdx];

			node.parentIdx = 0;
			node.fence = -1;
			leafNodes[tid] = node;
		}

		// Builds out the internal nodes of the LBVH
		__global__ void lbvhBuildInternalKernel(LBVH::node* nodes,
			int* flags, uint32_t const* mortonCodes, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs - 1) return;

			const ivec2 range = determineRange(mortonCodes, numObjs, tid);
			const int gamma = findSplit(mortonCodes, range.x, range.y);

			// Left and right children are neighbors to the split point
			// Check if there are leaf nodes, which are indexed behind the (numObj - 1) internal nodes
			ivec2 children(
				(range.x == gamma) ? gamma + numObjs - 1 : gamma,
				(range.y == gamma + 1) ? gamma + numObjs : gamma + 1
			);

			nodes[tid].leftIdx = children.x;
			nodes[tid].rightIdx = children.y;
			nodes[tid].fence = (tid == range.x) ? range.y : range.x;
			nodes[children.x].parentIdx = tid;
			nodes[children.y].parentIdx = tid;

			// Reset flags for mergeUpKernel
			flags[tid] = 0;
		}

		// Copy over leaf node aabbs if they have changed
		__global__ void copyAABBKernel(LBVH::node* leafNodes,
			Bound<3, float>* leafAABBs, Bound<3, float>* objAABBs, int* flags, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			leafAABBs[tid] = objAABBs[leafNodes[tid].objIdx];
			if (tid < numObjs - 1)		// Reset internal flags because we need to refit after this
				flags[tid] = 0;
		}

		// Refits the AABBs of the internal nodes
		__global__ void mergeUpKernel(LBVH::node* nodes, Bound<3, float>* aabbs,
			int* flags, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			// Keep track of the maximum stack size required for DFS
			// Assuming full exploration and always pushing the left child first.
			int depth = 1;

			int lastID = tid + numObjs - 1;
			int parent = nodes[lastID].parentIdx;
			Bound<3, float> last = aabbs[lastID];
			while (true) {
				// Exit if we are the first thread here
				int otherDepth = atomicOr(flags + parent, depth);
				if (!otherDepth) return;

				LBVH::node node = nodes[parent];
				int sibling;
				if (node.leftIdx == lastID) {					// We were the left child
					sibling = node.rightIdx;
					depth = glm::max(depth, otherDepth + 1);
				}
				else {											// We were the right child
					sibling = node.leftIdx;
					depth = glm::max(depth + 1, otherDepth);
				}

				// Ensure memory coherency before we read.
				__threadfence();

				last.absorb(aabbs[sibling]);
				aabbs[parent] = last;
				if (!parent) {			// We've reached the root.
					flags[0] = depth;	// Only the one lucky thread gets to finish up.
					return;
				}
				lastID = parent;
				parent = node.parentIdx;
			}
		}

		// We do 128 wide blocks which gets us 8KB of shared memory per block on compute 8.6. 
		// Realistically, though, we are limited by the number of registers so we can overallocate shared memory.
		constexpr int MAX_RES_PER_BLOCK = 8 * 128;

		// Query the LBVH for overlapping objects
		// Overcomplicated because of shared memory buffering
		template<bool IGNORE_SELF, int STACK_SIZE>
		__global__ void lbvhQueryKernel(ivec2* res, int* resCounter, int maxRes,
			LBVH::node const* nodes, Bound<3, float> const* aabbs,
			LBVH::node const* queryNodes, Bound<3, float> const* queryAabbs, const int numQueries) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			bool active = tid < numQueries;

			Bound<3, float> queryAABB;
			int objIdx;
			if (active) {
				queryAABB = queryAabbs[tid];
				objIdx = queryNodes[tid].objIdx;
			}

			__shared__ ivec2 sharedRes[MAX_RES_PER_BLOCK];
			__shared__ int sharedCounter;		// How many results are cached in shared memory
			__shared__ int sharedGlobalIdx;		// Where to write in global memory
			if (threadIdx.x == 0)
				sharedCounter = 0;

			int stack[STACK_SIZE];				// We only need 32 nodes for a 2^31 tree
			int* stackPtr = stack;
			*(stackPtr++) = 0;					// Push

			while (true) {
				__syncthreads();

				if (active)
					while (stackPtr != stack) {
						int nodeIdx = *(--stackPtr);	// Pop
						auto node = nodes[nodeIdx];

						if (node.isLeaf()) {
							if (IGNORE_SELF)
								if (nodeIdx <= tid + numQueries - 1) continue;
							// Add to shared memory
							int sIdx = atomicAdd(&sharedCounter, 1);
							if (sIdx >= MAX_RES_PER_BLOCK) {
								// We cannot sync here so we push the node back on the stack and wait
								*(stackPtr++) = nodeIdx;
								break;
							}
							sharedRes[sIdx] = ivec2(node.objIdx, objIdx);
						}
						else {
							// Ignore duplicate and self intersections
							if (IGNORE_SELF)
								if (glm::max(nodeIdx, node.fence) <= tid) continue;

							// Internal node
							const Bound<3, float> leftAABB = aabbs[node.leftIdx];
							if (leftAABB.intersects(queryAABB))
								*(stackPtr++) = node.leftIdx;	// Push

							const Bound<3, float> rightAABB = aabbs[node.rightIdx];
							if (rightAABB.intersects(queryAABB))
								*(stackPtr++) = node.rightIdx;	// Push
						}
					}

				// Flush whatever we have
				__syncthreads();
				int totalRes = glm::min(sharedCounter, MAX_RES_PER_BLOCK);

				if (threadIdx.x == 0)
					sharedGlobalIdx = atomicAdd(resCounter, totalRes);

				__syncthreads();

				// Make sure we dont write out of bounds
				const int globalIdx = sharedGlobalIdx;

				if (globalIdx >= maxRes || !totalRes) return;	// Out of memory for results.
				if (threadIdx.x == 0) sharedCounter = 0;

				// If we got here with a half full buffer, we are done.
				bool done = totalRes < MAX_RES_PER_BLOCK;
				// If we are about to run out of memory, we are done.
				if (totalRes > maxRes - globalIdx) {
					totalRes = maxRes - globalIdx;
					done = true;
				}

				// Copy full blocks
				int fullBlocks = (totalRes - 1) / (int)blockDim.x;
				for (int i = 0; i < fullBlocks; i++) {
					int idx = i * blockDim.x + threadIdx.x;
					res[globalIdx + idx] = sharedRes[idx];
				}

				// Copy the rest
				int idx = fullBlocks * blockDim.x + threadIdx.x;
				if (idx < totalRes) res[globalIdx + idx] = sharedRes[idx];

				// Break if every thread is done.
				if (done) break;
			}
		}

		// We are primarily limited by the number of registers, so we always call the kernel with just enough stack space.
		// This gives another ~15% performance boost for queries.
		template<bool IGNORE_SELF>
		void launchQueryKernel(ivec2* res, int* resCounter, int maxRes,
			LBVH::node const* nodes, Bound<3, float> const* aabbs,
			LBVH::node const* queryNodes, Bound<3, float> const* queryAabbs, const int numQueries, int stackSize) {

			// This is a bit ugly but we want to compile the kernel for all stack sizes.
#define DISPATCH_QUERY(N) case N: lbvhQueryKernel<IGNORE_SELF, N> << <(numQueries + 127) / 128, 128 >> > (res, resCounter, maxRes, nodes, aabbs, queryNodes, queryAabbs, numQueries); break;
			switch (stackSize) {
			default:
				DISPATCH_QUERY(32); DISPATCH_QUERY(31); DISPATCH_QUERY(30); DISPATCH_QUERY(29); DISPATCH_QUERY(28); DISPATCH_QUERY(27); DISPATCH_QUERY(26); DISPATCH_QUERY(25);
				DISPATCH_QUERY(24); DISPATCH_QUERY(23); DISPATCH_QUERY(22); DISPATCH_QUERY(21); DISPATCH_QUERY(20); DISPATCH_QUERY(19); DISPATCH_QUERY(18); DISPATCH_QUERY(17);
				DISPATCH_QUERY(16); DISPATCH_QUERY(15); DISPATCH_QUERY(14); DISPATCH_QUERY(13); DISPATCH_QUERY(12); DISPATCH_QUERY(11); DISPATCH_QUERY(10); DISPATCH_QUERY(9);
				DISPATCH_QUERY(8); DISPATCH_QUERY(7); DISPATCH_QUERY(6); DISPATCH_QUERY(5); DISPATCH_QUERY(4); DISPATCH_QUERY(3); DISPATCH_QUERY(2); DISPATCH_QUERY(1);
			}
#undef DISPATCH_QUERY
		}
	}

#pragma endregion
#pragma region LBVH

	void LBVH::refitNoCopy() {
		// Go through and merge all the aabbs up from the leaf nodes
		LBVHKernels::mergeUpKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(d_nodes.data()),
			thrust::raw_pointer_cast(d_aabbs.data()),
			thrust::raw_pointer_cast(d_flags.data()), numObjs);
	}

	void LBVH::refit() {
		// Copy over leaf node aabbs
		LBVHKernels::copyAABBKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(d_nodes.data()) + numObjs - 1,
			thrust::raw_pointer_cast(d_aabbs.data()) + numObjs - 1,
			thrust::raw_pointer_cast(d_objs),
			thrust::raw_pointer_cast(d_flags.data()), numObjs);

		refitNoCopy();
		checkCudaErrors(cudaGetLastError());
	}

	void LBVH::compute(aabb* devicePtr, size_t size) {
		d_objs = thrust::device_ptr<aabb>(devicePtr);
		numObjs = size;

		const unsigned int numInternalNodes = numObjs - 1;	// Total number of internal nodes
		const unsigned int numNodes = numObjs * 2 - 1;		// Total number of nodes

		d_morton.resize(numObjs);
		d_objIDs.resize(numObjs);
		d_nodes.resize(numNodes);
		d_aabbs.resize(numNodes);
		d_flags.resize(numInternalNodes);

		// Compute the bounding box for the whole scene so we can assign morton codes
		aabb wholeAABB;
		wholeAABB = thrust::reduce(
			d_objs, d_objs + numObjs, wholeAABB,
			[] __host__ __device__(const aabb & lhs, const aabb & rhs) {
			auto b = lhs;
			b.absorb(rhs);
			return b;
		});

		// Compute morton codes. These don't have to be unique here.
		LBVHKernels::mortonKernel<float> << <(numObjs + 255) / 256, 256 >> > (
			devicePtr, thrust::raw_pointer_cast(d_morton.data()),
			thrust::raw_pointer_cast(d_objIDs.data()), wholeAABB, numObjs);

		// Sort morton codes
		thrust::stable_sort_by_key(d_morton.begin(), d_morton.end(), d_objIDs.begin());

		// Build out the leaf nodes
		LBVHKernels::lbvhBuildLeafKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(d_nodes.data()) + numObjs - 1,
			thrust::raw_pointer_cast(d_aabbs.data()) + numObjs - 1,
			devicePtr, thrust::raw_pointer_cast(d_objIDs.data()), numObjs);

		// Build out the internal nodes
		LBVHKernels::lbvhBuildInternalKernel << <(numInternalNodes + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(d_nodes.data()),
			thrust::raw_pointer_cast(d_flags.data()),
			thrust::raw_pointer_cast(d_morton.data()), numObjs);

		refitNoCopy();
		maxStackSize = d_flags[0];		// Save max depth for query invocation

		checkCudaErrors(cudaGetLastError());
	}

	size_t LBVH::query(ivec2* d_res, size_t resSize, LBVH* other) const {
		// Borrow the flags array for the counter
		int* d_counter = (int*)thrust::raw_pointer_cast(d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		// Query the LBVH
		const int numQuery = other->numObjs;
		LBVHKernels::launchQueryKernel<false>(
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(d_nodes.data()),
			thrust::raw_pointer_cast(d_aabbs.data()),
			thrust::raw_pointer_cast(other->d_nodes.data()) + other->numObjs - 1,
			thrust::raw_pointer_cast(other->d_aabbs.data()) + other->numObjs - 1,
			numQuery, maxStackSize
		);

		checkCudaErrors(cudaGetLastError());
		return std::min((size_t)d_flags[0], resSize);
	}

	size_t LBVH::query(ivec2* d_res, size_t resSize) const {
		// Borrow the flags array for the counter
		int* d_counter = (int*)thrust::raw_pointer_cast(d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		// Query the LBVH
		const int numQuery = numObjs;
		LBVHKernels::launchQueryKernel<true>(
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(d_nodes.data()),
			thrust::raw_pointer_cast(d_aabbs.data()),
			thrust::raw_pointer_cast(d_nodes.data()) + numObjs - 1,
			thrust::raw_pointer_cast(d_aabbs.data()) + numObjs - 1,
			numQuery, maxStackSize
		);

		checkCudaErrors(cudaGetLastError());
		return std::min((size_t)d_flags[0], resSize);
	}

#pragma endregion
#pragma region testing

	Kitten::Bound<3, float> lbvhCheckAABBMerge(
		thrust::host_vector<LBVH::node>& nodes,
		thrust::host_vector<Kitten::Bound<3, float>>& aabbs,
		int idx) {

		auto aabb = aabbs[idx];
		auto node = nodes[idx];
		if (node.isLeaf()) return aabb;

		auto left = lbvhCheckAABBMerge(nodes, aabbs, node.leftIdx);
		auto right = lbvhCheckAABBMerge(nodes, aabbs, node.rightIdx);
		left.absorb(right);

		// Check if they match
		if (length2(left.min - aabb.min) > 1e-7f || length2(left.max - aabb.max) > 1e-7f) {
			printf("Error: AABB mismatch node %d\n", idx);
			printf("Expected:\n");
			Kitten::print(left.min);
			Kitten::print(left.max);
			printf("Found:\n");
			Kitten::print(aabb.min);
			Kitten::print(aabb.max);
		}

		return left;
	}

	ivec2 lbvhCheckIndexMerge(
		thrust::host_vector<LBVH::node>& nodes,
		int idx, int numObjs) {

		auto node = nodes[idx];
		if (node.isLeaf()) return ivec2(idx - numObjs + 1);
		ivec2 range(idx, node.fence);
		if (range.y < range.x) std::swap(range.x, range.y);

		auto left = lbvhCheckIndexMerge(nodes, node.leftIdx, numObjs);
		auto right = lbvhCheckIndexMerge(nodes, node.rightIdx, numObjs);
		left.x = std::min(left.x, right.x);
		left.y = std::max(left.y, right.y);

		// Check if they match
		if (left != range)
			printf("Error: Index range mismatch\n");

		return left;
	}

	void LBVH::bvhSelfCheck() const {
		printf("\nLBVH self check...\n");

		// Get nodes
		thrust::host_vector<node> nodes(d_nodes.begin(), d_nodes.end());
		thrust::host_vector<aabb> aabbs(d_aabbs.begin(), d_aabbs.end());
		thrust::host_vector<uint64_t> morton(d_morton.begin(), d_morton.end());

		// Check sizes
		if (nodes.size() != numObjs * 2 - 1 || aabbs.size() != numObjs * 2 - 1 || morton.size() != numObjs)
			printf("Error: Incorrect memory sizes\n");

		// Check morton codes
		for (size_t i = 1; i < numObjs; i++)
			if (morton[i - 1] > morton[i])
				printf("Bad morton code ordering\n");

		// Check leaf nodes
		for (size_t i = 0; i < numObjs; i++) {
			auto node = nodes[i + numObjs - 1];
			if (!node.isLeaf())
				printf("Error: Leaf node is not a leaf\n");
		}

		// Check that all children have the correct parent
		for (size_t i = 0; i < numObjs - 1; i++) {
			auto node = nodes[i];
			if (node.isLeaf())
				printf("Error: Internal node is a leaf\n");
			if (nodes[node.leftIdx].parentIdx != i || nodes[node.rightIdx].parentIdx != i)
				printf("Error: Child node has incorrect parent\n");
		}

		// Check that all nodes are accessible from the root
		std::vector<int> stack;
		int numVisited = 0;
		int maxSize = 0;
		stack.push_back(0);
		while (stack.size()) {
			int idx = stack.back();
			stack.pop_back();
			numVisited++;
			auto node = nodes[idx];
			if (!node.isLeaf()) {
				stack.push_back(node.leftIdx);
				stack.push_back(node.rightIdx);
			}
			maxSize = std::max(maxSize, (int)stack.size());
		}
		if (numVisited != numObjs * 2 - 1)
			printf("Error: Not all nodes are accessible from the root. Only %d/%d nodes are found.\n", numVisited, 2 * numObjs - 1);
		if (maxStackSize != maxSize)
			printf("Error: Max stack size mismatch. Stored %d, found %d\n", maxStackSize, maxSize);

		// Print all nodes. Useful for debugging
		if (false)
			for (size_t i = 0; i < nodes.size(); i++) {
				printf("\n--node %d--\n", i);
				printf("parent: %d\n", nodes[i].parentIdx);
				printf("left: %d\n", nodes[i].leftIdx);
				printf("right: %d\n", nodes[i].rightIdx);
				printf("fence: %d\n", nodes[i].fence);
				Kitten::print(aabbs[i].min);
				Kitten::print(aabbs[i].max);
			}

		// Check merging of indices and aabbs
		lbvhCheckIndexMerge(nodes, 0, numObjs);
		lbvhCheckAABBMerge(nodes, aabbs, 0);

		printf("Num nodes: %d\n", nodes.size());
		printf("Max stack size: %d\n", maxStackSize);
		printf("LBVH self check complete.\n\n");
	}

	// Tests the LBVH
	void testLBVH() {
		const int N = 100000;
		const float R = 0.001f;

		printf("Generating Data...\n");
		vector<Kitten::Bound<3, float>> points(N);

		srand(1);
		for (size_t i = 0; i < N; i++) {
			Kitten::Bound<3, float> b(vec3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX));
			b.pad(R);
			points[i] = b;
		}

		thrust::device_vector<Kitten::Bound<3, float>> d_points(points.begin(), points.end());
		thrust::device_vector<ivec2> d_res(100 * N);
		cudaDeviceSynchronize();

		// Build BVH
		LBVH bvh;
		printf("Building LBVH...\n");
		bvh.compute(thrust::raw_pointer_cast(d_points.data()), N);
		cudaDeviceSynchronize();

		// Query BVH
		printf("Querying LBVH...\n");
		int numCols = bvh.query(thrust::raw_pointer_cast(d_res.data()), d_res.size());

		// Print results
		printf("Getting results...\n");
		thrust::host_vector<ivec2> res(d_res.begin(), d_res.begin() + numCols);

		printf("%d collision pairs found on GPU.\n", res.size());
		// printf("GPU:\n");
		// for (size_t i = 0; i < res.size(); i++)
		// 	printf("%d %d\n", res[i].x, res[i].y);

		// Brute force compute the same result
		std::unordered_set<ivec2> resSet;
		bool good = true;

		for (size_t i = 0; i < res.size(); i++) {
			ivec2 a = res[i];
			if (a.x > a.y) std::swap(a.x, a.y);
			if (!resSet.insert(a).second) {
				printf("Error: Duplicate result\n");
				good = false;
			}
		}

		int numCPUFound = 0;
		printf("\nRunning brute force CPU collision detection...\n");
#pragma omp parallel for
		for (int i = 0; i < N; i++)
			for (int j = i + 1; j < N; j++)
				if (points[i].intersects(points[j])) {
#pragma omp atomic
					numCPUFound++;
					if (resSet.find(ivec2(i, j)) == resSet.end()) {
						printf("Error: CPU result %d %d not found in GPU result.\n", i, j);
						good = false;
					}
				}

		if (numCPUFound != res.size()) {
			printf("Error: CPU and GPU results do not match\n");
			good = false;
		}

		printf("%d collision pairs found on CPU.\n", numCPUFound);
		printf(good ? "CPU and GPU results match.\n" : "CPU and GPU results MISMATCH!\n");

		bvh.bvhSelfCheck();
	}

#pragma endregion

}