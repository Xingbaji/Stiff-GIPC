//
// mlbvh.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _MLBVH_CUH_
#define _MLBVH_CUH_
#include <cstdint>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

struct AABB
{
  public:
    double3             upper;
    double3             lower;
    __host__ __device__ AABB();
    __host__ __device__ void combines(const double& x, const double& y, const double& z);
    __host__ __device__ void    combines(const double& x,
                                         const double& y,
                                         const double& z,
                                         const double& xx,
                                         const double& yy,
                                         const double& zz);
    __host__ __device__ void    combines(const AABB& aabb);
    __host__ __device__ double3 center();
};

struct Node
{
  public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class lbvh
{
  public:
    uint32_t  vert_number;
    double3*  _vertexes;
    AABB*     _bvs;
    AABB*     _tempLeafBox;
    Node*     _nodes;
    uint64_t* _MChash;
    uint32_t* _indices;
    int4*     _collisionPair;
    int4*     _ccd_collisionPair;
    uint32_t* _cpNum;
    int*      _MatIndex;
    uint32_t* _flags;
    AABB      scene;
    int*      _btype;
    int*      _bodyId;

  public:
    lbvh() {}
    ~lbvh();
    void MALLOC_DEVICE_MEM(const int& number);
    void FREE_DEVICE_MEM();
    //void Construct();
};


class lbvh_f : public lbvh
{
  public:
    uint32_t  face_number;
    uint3*    _faces;
    uint32_t* _surfVerts;

  public:
    void   init(int*       _bodyID,
                int*       _btype,
                double3*   _mVerts,
                uint3*     _mFaces,
                uint32_t*  _mSurfVert,
                int4*      _mCollisonPairs,
                int4*      _ccd_mCollisonPairs,
                uint32_t*  _mcpNum,
                int*       _mMatIndex,
                const int& faceNum,
                const int& vertNum);
    double Construct();
    AABB*  getSceneSize();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void   SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);
};

class lbvh_e : public lbvh
{
  public:
    double3* _rest_vertexes;
    uint32_t edge_number;
    uint2*   _edges;

  public:
    void   init(int*       _bodyID,
                int*       _btype,
                double3*   _mVerts,
                double3*   _rest_vertexes,
                uint2*     _mEdges,
                int4*      _mCollisonPairs,
                int4*      _ccd_mCollisonPairs,
                uint32_t*  _mcpNum,
                int*       _mMatIndex,
                const int& edgeNum,
                const int& vertNum);
    double Construct();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void   SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);
};

// ============================================================================
// Inline device functions for distance calculations
// These MUST be inlined for performance in CUDA separable compilation mode
// ============================================================================

#include "gpu_eigen_libs.cuh"

__device__ __forceinline__ void _d_PP(const double3& v0, const double3& v1, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__ __forceinline__ void _d_PT(const double3& v0,
                                      const double3& v1,
                                      const double3& v2,
                                      const double3& v3,
                                      double&        d)
{
    double3 b    = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1),
                                          __GEIGEN__::__minus(v3, v1));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ __forceinline__ void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)))
        / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__ __forceinline__ void _d_EE(const double3& v0,
                                      const double3& v1,
                                      const double3& v2,
                                      const double3& v3,
                                      double&        d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0),
                                          __GEIGEN__::__minus(v3, v2));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ __forceinline__ void _d_EEParallel(const double3& v0,
                                              const double3& v1,
                                              const double3& v2,
                                              const double3& v3,
                                              double&        d)
{
    double3 b = __GEIGEN__::__v_vec_cross(
        __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)),
        __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ __forceinline__ double _compute_epx(const double3& v0,
                                               const double3& v1,
                                               const double3& v2,
                                               const double3& v3)
{
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))
           * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

#endif