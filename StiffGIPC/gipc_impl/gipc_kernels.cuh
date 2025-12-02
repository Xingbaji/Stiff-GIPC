//
// gipc_kernels.cuh
// GIPC CUDA Kernel and Function Declarations
//
// This file declares all CUDA kernels and host helper functions
// used in the GIPC physics simulation.
//
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef GIPC_KERNELS_CUH
#define GIPC_KERNELS_CUH

#include <Eigen/Eigen>
#include "../gpu_eigen_libs.cuh"
#include "../FrictionUtils.cuh"
#include "../GIPC_PDerivative.cuh"
#include "../mlbvh.cuh"

// ============================================================================
// Common Constants
// ============================================================================

constexpr int gipc_default_threads = 256;
constexpr int GIPC_M12_Off = 16;  // 4x4 blocks for 12x12 matrices
constexpr int GIPC_M9_Off = 9;    // 3x3 blocks for 9x9 matrices
constexpr int GIPC_M6_Off = 4;    // 2x2 blocks for 6x6 matrices

// ============================================================================
// Host Helper Function Declarations
// ============================================================================

/// Step vertices forward by alpha * moveDir
void stepForward(double3*       o_vertexes,
                 const double3* vertexes,
                 const double3* moveDir,
                 const int*     bType,
                 double         alpha,
                 bool           moveBoundary,
                 int            number);

/// Calculate kinetic gradient
void calKineticGradient(double3* _vertexes, 
                        double3* _xTilta, 
                        double3* _gradient, 
                        double*  _masses, 
                        int      numbers);

/// Calculate minimum movement norm
double calcMinMovement(const double3* _moveDir, 
                       double*        _queue, 
                       const int&     number);

/// Calculate FEM gradient and Hessian
void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                    const double3*          vertexes,
                                    const uint4*            tetrahedras,
                                    const double*           volume,
                                    double3*                gradient,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    triplet_rows,
                                    int*                    triplet_cols,
                                    double                  lenRate,
                                    double                  volRate,
                                    double                  IPC_dt,
                                    double*                 queue,
                                    int                     tetrahedraNum,
                                    int                     vertNum,
                                    int                     offset);

// ============================================================================
// Barrier (IPC) Kernel Declarations
// Defined in gipc_impl/gipc_barrier_kernels.cu
// ============================================================================

__global__ void _calBarrierHessian(const double3*   _vertexes,
                                   const double3*   _rest_vertexes,
                                   const int4*      _collisionPair,
                                   Eigen::Matrix3d* triplet_values,
                                   int*             row_ids,
                                   int*             col_ids,
                                   uint32_t*        _cpNum,
                                   int*             matIndex,
                                   double           dHat,
                                   double           Kappa,
                                   int              offset4,
                                   int              offset3,
                                   int              offset2,
                                   int              number);

__global__ void _calBarrierGradientAndHessian(const double3*   _vertexes,
                                              const double3*   _rest_vertexes,
                                              const int4*      _collisionPair,
                                              double3*         _gradient,
                                              Eigen::Matrix3d* triplet_values,
                                              int*             row_ids,
                                              int*             col_ids,
                                              uint32_t*        _cpNum,
                                              int*             matIndex,
                                              double           dHat,
                                              double           Kappa,
                                              int              offset4,
                                              int              offset3,
                                              int              offset2,
                                              int              number);

__global__ void _calBarrierGradient(const double3* _vertexes,
                                    const double3* _rest_vertexes,
                                    const int4*    _collisionPair,
                                    double3*       _gradient,
                                    double         dHat,
                                    double         Kappa,
                                    int            number);

// ============================================================================
// Friction Kernel Declarations
// Defined in gipc_impl/gipc_friction_kernels.cu
// ============================================================================

__global__ void _calFrictionHessian_gd(const double3*   _vertexes,
                                       const double3*   _o_vertexes,
                                       const double3*   _normal,
                                       const uint32_t*  _last_collisionPair_gd,
                                       Eigen::Matrix3d* triplet_values,
                                       int*             row_ids,
                                       int*             col_ids,
                                       int              number,
                                       double           dt,
                                       double           eps2,
                                       double*          lastH,
                                       int              global_offset,
                                       double           coef);

__global__ void _calFrictionHessian(const double3*          _vertexes,
                                    const double3*          _o_vertexes,
                                    const int4*             _last_collisionPair,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    uint32_t*               _cpNum,
                                    int                     number,
                                    double                  dt,
                                    double2*                distCoord,
                                    __GEIGEN__::Matrix3x2d* tanBasis,
                                    double                  eps2,
                                    double*                 lastH,
                                    double                  coef,
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2);

__global__ void _calFrictionGradient_gd(const double3*  _vertexes,
                                        const double3*  _o_vertexes,
                                        const double3*  _normal,
                                        const uint32_t* _last_collisionPair_gd,
                                        double3*        _gradient,
                                        int             number,
                                        double          dt,
                                        double          eps2,
                                        double*         lastH,
                                        double          coef);

__global__ void _calFrictionGradient(const double3*          _vertexes,
                                     const double3*          _o_vertexes,
                                     const int4*             _last_collisionPair,
                                     double3*                _gradient,
                                     int                     number,
                                     double                  dt,
                                     double2*                distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef);

__global__ void _getFrictionEnergy_Reduction_3D(double*                       squeue,
                                                const double3*                vertexes,
                                                const double3*                o_vertexes,
                                                const int4*                   _collisionPair,
                                                int                           cpNum,
                                                double                        dt,
                                                const double2*                distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double*                 lastH,
                                                double                        fricDHat,
                                                double                        eps);

__global__ void _getFrictionEnergy_gd_Reduction_3D(double*         squeue,
                                                   const double3*  vertexes,
                                                   const double3*  o_vertexes,
                                                   const double3*  _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int             gpNum,
                                                   double          dt,
                                                   const double*   lastH,
                                                   double          eps);

__global__ void _calFrictionLastH_gd(const double3*    _vertexes,
                                     const double*     g_offset,
                                     const double3*    g_normal,
                                     const uint32_t*   _collisionPair_environment,
                                     double*           lambda_lastH_gd,
                                     uint32_t*         _collisionPair_last_gd,
                                     double            dHat,
                                     double            Kappa,
                                     int               number);

__global__ void _calFrictionLastH_DistAndTan(const double3*          _vertexes,
                                             const int4*             _collisionPair,
                                             double*                 lambda_lastH,
                                             double2*                distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*                   _collisionPair_last,
                                             double                  dHat,
                                             double                  Kappa,
                                             uint32_t*               _cpNum_last,
                                             int                     number);

// ============================================================================
// Ground Collision Kernel Declarations
// Defined in gipc_impl/gipc_ground_kernels.cu
// ============================================================================

__global__ void _GroundCollisionDetect(const double3*  vertexes,
                                       const uint32_t* surfVertIds,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       uint32_t* _environment_collisionPair,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       int       number);

__global__ void _getTotalForce(const double3* _force0, double3* _force, int number);

__global__ void _computeGroundGradientAndHessian(const double3* vertexes,
                                                 const double*  g_offset,
                                                 const double3* g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 double3*  gradient,
                                                 uint32_t* _gpNum,
                                                 Eigen::Matrix3d* triplet_values,
                                                 int*   row_ids,
                                                 int*   col_ids,
                                                 double dHat,
                                                 double Kappa,
                                                 int    global_offset,
                                                 int    number);

__global__ void _computeGroundGradient(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*  gradient,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       double    Kappa,
                                       int       number);

__global__ void _computeGroundCloseVal(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double    dTol,
                                       uint32_t* _closeConstraintID,
                                       double*   _closeConstraintVal,
                                       uint32_t* _close_gpNum,
                                       int       number);

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number);

__global__ void _checkGroundIntersection(const double3* vertexes,
                                         const double*  g_offset,
                                         const double3* g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int* _isIntersect,
                                         int  number);

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number);

__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number);

__global__ void _reduct_min_groundTimeStep_to_double(const double3* vertexes,
                                                     const uint32_t* surfVertIds,
                                                     const double*  g_offset,
                                                     const double3* g_normal,
                                                     const double3* moveDir,
                                                     double* minStepSizes,
                                                     double  slackness,
                                                     int     number);

// ============================================================================
// Energy Kernel Declarations
// Defined in gipc_impl/gipc_energy_kernels.cu
// ============================================================================

__global__ void _calKineticGradient(double3* _vertexes,
                                    double3* _xTilta,
                                    double3* _gradient,
                                    double*  _masses,
                                    int      numbers);

__global__ void _calKineticEnergy(double3* _vertexes,
                                  double3* _xTilta,
                                  double*  _energy,
                                  double*  _masses,
                                  int      numbers);

__global__ void _getKineticEnergy_Reduction_3D(double3* _vertexes,
                                               double3* _xTilta,
                                               double*  _energy,
                                               double*  _masses,
                                               int      number);

__global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double* volume,
                                           int           tetrahedraNum,
                                           double*       lenRate,
                                           double*       volRate);

__global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff,
                                                    double        strainRate);

__global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate);

__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff);

__global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum);

__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double motionRate,
                                                         double rate,
                                                         int    global_offset,
                                                         int    global_hessian_fem_offset,
                                                         int    number);

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               int             number);

__global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       int    number);

__global__ void _getDeltaEnergy_Reduction(double* squeue,
                                          const double3* b,
                                          const double3* dx,
                                          int vertexNum);

__global__ void __add_reduction(double* mem, int numbers);

// ============================================================================
// Reduction Kernel Declarations
// Defined in gipc_impl/gipc_reduction_kernels.cu
// ============================================================================

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim,
                                              double* _double1Dim,
                                              int number);

__global__ void _reduct_min_double(double* _double1Dim, int number);

__global__ void _reduct_M_double2(double2* _double2Dim, int number);

__global__ void _reduct_max_double(double* _double1Dim, int number);

__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number);

__global__ void _reduct_double3Dot_to_double(const double3* A,
                                             const double3* B,
                                             double* D,
                                             int number);

__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number);

__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number);

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number);

__global__ void _reduct_max_cfl_to_double(const double3* moveDir,
                                          double*        max_double_val,
                                          uint32_t*      mSVI,
                                          int            number);

__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4*   tetrahedra,
                                                        const double3* moveDir,
                                                        double*        minStepSizes,
                                                        double         slackness,
                                                        double         errorRate,
                                                        int            number);

__global__ void _reduct_min_selfTimeStep_to_double(const double3* vertexes,
                                                   const int4*    _ccd_collisionPairs,
                                                   const double3* moveDir,
                                                   double*        minStepSizes,
                                                   double         slackness,
                                                   int            number);

// ============================================================================
// Simulation Step Kernel Declarations
// Defined in gipc_impl/gipc_simulation_kernels.cu
// ============================================================================

__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers);

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers);

__global__ void _updateBoundary(double3* _vertexes,
                                int*     _btype,
                                double3* _moveDir,
                                double   ipc_dt,
                                int      numbers);

__global__ void _updateBoundary2(int* _btype,
                                 __GEIGEN__::Matrix3x3d* _constraints,
                                 int numbers);

__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers,
                                       int      frameid);

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               double   ipc_dt,
                               double   rate,
                               int      numbers);

__global__ void _updateSurfaces(uint32_t* sortIndex,
                                uint3* _faces,
                                int _offset_num,
                                int numbers);

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers);

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers);

__global__ void _updateEdges(uint32_t* sortIndex,
                             uint2* _edges,
                             int _offset_num,
                             int numbers);

__global__ void _updateTriEdges_adjVerts(uint32_t* sortIndex,
                                         uint2* _edges,
                                         uint2* _adj_verts,
                                         int _offset_num,
                                         int numbers);

__global__ void _updateSurfVerts(uint32_t* sortIndex,
                                 uint32_t* _sVerts,
                                 int _offset_num,
                                 int numbers);

__global__ void _edgeTriIntersectionQuery(const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            number);

#endif  // GIPC_KERNELS_CUH
