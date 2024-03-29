#pragma once
// Jerry Hsu, 2021

#define KITTEN_FUNC_DECL 

#if __has_include("cuda_runtime.h")
#pragma nv_diag_suppress esa_on_defaulted_function_ignored
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#undef KITTEN_FUNC_DECL
#define KITTEN_FUNC_DECL __device__ __host__

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/norm.hpp>

#include <string>
#include <functional>
#include <tuple>

namespace Kitten {
	using namespace glm;
	using namespace std;

	inline void print(int x, const char* format = "%d") {
		printf(format, x);
	}

	KITTEN_FUNC_DECL inline void printDiv(const char* label = nullptr) {
		if (!label) label = "div";
		printf("\n----%s----\n", label);
	}

	KITTEN_FUNC_DECL inline void print(float x, const char* format = "%f\n") {
		printf(format, x);
	}

	KITTEN_FUNC_DECL inline void print(double x, const char* format = "%f\n") {
		printf(format, x);
	}

	template <int c, int r, typename T>
	KITTEN_FUNC_DECL void print(mat<c, r, T, defaultp> m, const char* format = "%.4f") {
		for (int i = 0; i < r; i++) {
			printf(i == 0 ? "{{" : " {");

			for (int j = 0; j < c; j++) {
				printf(format, m[j][i]);
				if (j != c - 1) printf(", ");
			}

			printf(i == r - 1 ? "}}\n" : "}\n");
		}
	}

	template <int s, typename T>
	KITTEN_FUNC_DECL void print(vec<s, T, defaultp> v, const char* format = "%.4f") {
		printf("{");
		for (int i = 0; i < s; i++) {
			printf(format, v[i]);
			if (i != s - 1) printf(", ");
		}
		printf("}\n");
	}

	template <int s>
	KITTEN_FUNC_DECL void print(vec<s, int, defaultp> v, const char* format = "%d") {
		printf("{");
		for (int i = 0; i < s; i++) {
			printf(format, v[i]);
			if (i != s - 1) printf(", ");
		}
		printf("}\n");
	}

	KITTEN_FUNC_DECL inline uint32_t mortonExpandBits(uint32_t v) {
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	KITTEN_FUNC_DECL inline uint32_t getMorton(ivec3 cell) {
		const uint32_t xx = mortonExpandBits(cell.x);
		const uint32_t yy = mortonExpandBits(cell.y);
		const uint32_t zz = mortonExpandBits(cell.z);
		return xx * 4 + yy * 2 + zz;
	}
}

namespace std {
	namespace {

		// Code from boost
		// Reciprocal of the golden ratio helps spread entropy
		//     and handles duplicates.
		// See Mike Seymour in magic-numbers-in-boosthash-combine:
		//     http://stackoverflow.com/questions/4948780

		template <class T>
		inline void hash_combine(std::size_t& seed, T const& v) {
			seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}

		// Recursive template code derived from Matthieu M.
		template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
		struct HashValueImpl {
			static void apply(size_t& seed, Tuple const& tuple) {
				HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
				hash_combine(seed, std::get<Index>(tuple));
			}
		};

		template <class Tuple>
		struct HashValueImpl<Tuple, 0> {
			static void apply(size_t& seed, Tuple const& tuple) {
				hash_combine(seed, std::get<0>(tuple));
			}
		};
	}

	template <int N>
	struct hash<glm::vec<N, int, glm::defaultp>> {
		std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
			size_t h = 0x9e3779b9;
			for (int i = 0; i < N; i++)
				h = v[i] ^ (h + 0x9e3779b9 + (v[i] << 6) + (v[i] >> 2));
			return h;
		}
	};

	template <int N>
	struct hash<glm::vec<N, glm::i64, glm::defaultp>> {
		std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
			size_t h = 0x9e3779b9;
			for (int i = 0; i < N; i++)
				h = v[i] ^ (h + 0x9e3779b9llu + (v[i] << 6) + (v[i] >> 2));
			return h;
		}
	};
};
