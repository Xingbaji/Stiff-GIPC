//
// gipc_simulation_kernels.cu
// GIPC - Simulation Step Kernels
//
// This file contains CUDA kernels for simulation step operations:
// - Vertex stepping and updates
// - Velocity updates
// - Boundary handling
// - Mesh topology updates
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <cuda_runtime.h>
#include "../gpu_eigen_libs.cuh"
#include "../mlbvh.cuh"
#include "gipc_device_functions.cuh"

// ============================================================================
// Vertex Update Kernels
// ============================================================================

__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        _vertexes[idx] =
            __GEIGEN__::__minus(_vertexesTemp[idx],
                                __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(btype[idx] == 0)
    {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else
    {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

// ============================================================================
// Boundary Update Kernels
// ============================================================================

__global__ void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == -1 || (_btype[idx]) == 1)
    {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__ void _updateBoundary2(int* _btype, __GEIGEN__::Matrix3x3d* _constraints, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == 1)
    {
        _btype[idx] = 0;
        __GEIGEN__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers,
                                       int      frameid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double                 massSum = 0;
    double                 angleX  = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(
        rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(
        rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    double mvl = -0.3 * ipc_dt * alpha;
    if((_btype[idx]) > 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
        }
    }
    if((_btype[idx]) < 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
        }
    }
}

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               double   ipc_dt,
                               double   rate,
                               int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double3 gravityDtSq = make_double3(0, 0, 0);
    if(_btype[idx] == 0)
    {
        gravityDtSq =
            __GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __GEIGEN__::__add(
        _o_vertexes[idx],
        __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt),
                          gravityDtSq));
}

// ============================================================================
// Mesh Topology Update Kernels
// ============================================================================

__global__ void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_faces[idx].x < _offset_num)
    {
        _faces[idx].x = sortIndex[_faces[idx].x];
    }
    if(_faces[idx].y < _offset_num)
    {
        _faces[idx].y = sortIndex[_faces[idx].y];
    }
    if(_faces[idx].z < _offset_num)
    {
        _faces[idx].z = sortIndex[_faces[idx].z];
    }
}

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int startId   = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum    = _neighborNum[idx];
    for(int i = 0; i < neiNum; i++)
    {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
}

__global__ void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
}

__global__ void _updateTriEdges_adjVerts(
    uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    if(_adj_verts[idx].x < _offset_num)
    {
        _adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
    }
    if(_adj_verts[idx].y < _offset_num)
    {
        _adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
    }
}

__global__ void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_sVerts[idx] < _offset_num)
    {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
}

// ============================================================================
// Intersection Query Kernel
// ============================================================================

__global__ void _edgeTriIntersectionQuery(const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    uint3 face = _faces[idx];

    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    double gapl = 0;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx   = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx   = _edge_nodes[node_id].right_idx;

        if(_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            return;
                        }
                }
            }
            else
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            return;
                        }
                }
            }
            else
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while(stack < stack_ptr);
}

