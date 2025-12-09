//
// device_fem_data.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "cuda_tools/cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num,
                                         const int& tetradedra_num,
                                         const int& triangle_num,
                                         const int& softNum,
                                         const int& tri_edgeNum,
                                         const int& bodyNum)
{
    int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
    CUDA_SAFE_CALL(cudaMalloc((void**)&vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&o_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&velocities, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rest_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp_double3Mem, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&xTilta, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&fb, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&totalForce, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&shape_grads, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&all_gradient, vertex_num * sizeof(double3)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&tetrahedras, tetradedra_num * sizeof(uint4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tempTetrahedras, tetradedra_num * sizeof(uint4)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edges, tri_edgeNum * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edge_adj_vertex, tri_edgeNum * sizeof(uint2)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&volum, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&masses, vertex_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lengthRate, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&volumeRate, tetradedra_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&tempDouble, maxNumbers * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&BoundaryType, vertex_num * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemset(BoundaryType, 0, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&DmInverses,
                              tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&targetIndex, softNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&targetVert, softNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triDmInverses,
                              triangle_num * sizeof(__GEIGEN__::Matrix2x2d)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&area, triangle_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triangles, triangle_num * sizeof(uint4)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&body_id_to_boundary_type, bodyNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&point_id_to_body_id, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tet_id_to_body_id, tetradedra_num * sizeof(int)));
}

device_TetraData::~device_TetraData()
{
    FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(vertexes));
    CUDA_SAFE_CALL(cudaFree(o_vertexes));
    CUDA_SAFE_CALL(cudaFree(temp_double3Mem));
    CUDA_SAFE_CALL(cudaFree(velocities));
    CUDA_SAFE_CALL(cudaFree(rest_vertexes));
    CUDA_SAFE_CALL(cudaFree(xTilta));
    CUDA_SAFE_CALL(cudaFree(fb));
    CUDA_SAFE_CALL(cudaFree(shape_grads));
    CUDA_SAFE_CALL(cudaFree(all_gradient));
    CUDA_SAFE_CALL(cudaFree(tetrahedras));
    CUDA_SAFE_CALL(cudaFree(tempTetrahedras));
    CUDA_SAFE_CALL(cudaFree(volum));
    CUDA_SAFE_CALL(cudaFree(masses));
    CUDA_SAFE_CALL(cudaFree(lengthRate));
    CUDA_SAFE_CALL(cudaFree(volumeRate));
    CUDA_SAFE_CALL(cudaFree(DmInverses));
    CUDA_SAFE_CALL(cudaFree(tempDouble));
    CUDA_SAFE_CALL(cudaFree(BoundaryType));

    CUDA_SAFE_CALL(cudaFree(totalForce));
    CUDA_SAFE_CALL(cudaFree(targetIndex));
    CUDA_SAFE_CALL(cudaFree(targetVert));
    CUDA_SAFE_CALL(cudaFree(triDmInverses));
    CUDA_SAFE_CALL(cudaFree(area));
    CUDA_SAFE_CALL(cudaFree(triangles));

    CUDA_SAFE_CALL(cudaFree(tri_edges));
    CUDA_SAFE_CALL(cudaFree(tri_edge_adj_vertex));

    CUDA_SAFE_CALL(cudaFree(body_id_to_boundary_type));
    CUDA_SAFE_CALL(cudaFree(point_id_to_body_id));
    CUDA_SAFE_CALL(cudaFree(tet_id_to_body_id));

}
