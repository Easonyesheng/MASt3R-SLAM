#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <cuda/std/limits>

#define BLOCK 16

// -----------------------------------------------------------------------------
// This file implements two CUDA kernels used by the matching frontend:
//
// 1) iter_proj_kernel:
//    Given a normalized 3D direction for each point, find the best pixel (u,v)
//    on a "ray image" by minimizing ||r(u,v) - x_hat||^2 with a tiny
//    2x2 Levenberg-Marquardt update per point.
//
// 2) refine_matches_kernel:
//    Optional local descriptor refinement around the projected pixel.
//
// Tensor layout conventions:
// - D11: [B, H, W, F]      descriptor map of reference image
// - D21: [B, N, F]         descriptor for query points (N = H*W)
// - p1 : [B, N, 2]         integer pixel coords (u,v)
// - rays_img_with_grad: [B, H, W, 9] = [ray_xyz, gx_xyz, gy_xyz]
// -----------------------------------------------------------------------------

__forceinline__ __device__ bool inside_image(int u, int v, int W, int H) {
  return v >= 0 && v < H && u >= 0 && u < W;
}

__forceinline__ __device__ void clamp(float& x, const float min, const float max) {
  x = fmin(fmax(x, min), max);
}

template <typename scalar_t>
__global__ void refine_matches_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> D11,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> D21,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1,
    torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1_new,
    const int radius,
    const int dilation_max
    )
{
  // Grid mapping:
  // - blockIdx.y = batch index b
  // - blockIdx.x/threadIdx.x enumerate points n inside that batch
  // batch index
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t b = blockIdx.y;

  const int h = D11.size(1);     // image height
  const int w = D11.size(2);     // image width
  const int fdim = D11.size(3);  // descriptor channel dimension

  // Get pixel and its features
  long u0 = p1[b][n][0];  // current predicted x pixel for point n
  long v0 = p1[b][n][1];  // current predicted y pixel for point n

  scalar_t max_score = ::cuda::std::numeric_limits<scalar_t>::min();  // best dot-product score so far
  long u_new = u0;  // best pixel x found in current dilation stage
  long v_new = v0;  // best pixel y found in current dilation stage

  // Coarse-to-fine local search:
  // d = dilation_max ... 1
  // For each dilation, inspect a square neighborhood with spacing d, choose the
  // best cosine-like score (dot product of descriptors), and recenter.
  for (int d=dilation_max; d>0; d--) {
    const int rd = radius*d;   // actual search radius under dilation d
    const int diam = 2*rd + 1; // side length of search window
    for (int i=0; i<diam; i+=d) {
      for (int j=0; j<diam; j+=d) {
        const long u = u0 - rd + i; // candidate x in window
        const long v = v0 - rd + j; // candidate y in window

        if (inside_image(u, v, w, h)) {
          scalar_t score = 0.0; // descriptor similarity at candidate pixel
          for (int k=0; k<fdim; k++) {
            score += D21[b][n][k] * D11[b][v][u][k]; // dot-product accumulation
          }

          if (score > max_score) {
            max_score = score; // update best similarity
            u_new = u;         // update best x
            v_new = v;         // update best y
          }
    
        }
      }
    }
    // Update where search is centered from previous update
    u0 = u_new; // recenter window for next (finer) dilation
    v0 = v_new; // recenter window for next (finer) dilation
  }

  p1_new[b][n][0] = u_new; // write refined x
  p1_new[b][n][1] = v_new; // write refined y
}


std::vector<torch::Tensor> refine_matches_cuda(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    const int radius,
    const int dilation)
{
  // Host wrapper:
  // launches one CUDA thread per (batch, point).
  const auto batch_size = p1.size(0);
  const auto n = p1.size(1);

  const dim3 blocks((n + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK);

  auto opts = p1.options();
  torch::Tensor p1_new = torch::zeros(
    {batch_size, n, 2}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(D11.type(), "refine_matches_kernel", ([&] {
    refine_matches_kernel<scalar_t><<<blocks, threads>>>(
      D11.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      D21.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      p1.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
      p1_new.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
      radius,
      dilation
    );
   }));

  return {p1_new};

}


__global__ void iter_proj_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> rays_img,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> pts_3d_norm,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_init,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_new,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> converged,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh
    )
{
  // Grid mapping:
  // - blockIdx.y = batch index b
  // - blockIdx.x/threadIdx.x enumerate points n for that edge/batch
  // batch index
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t b = blockIdx.y;

  const int h = rays_img.size(1); // image height
  const int w = rays_img.size(2); // image width
  const int c = rays_img.size(3); // channel count (=9: ray xyz + gx xyz + gy xyz)

  // Get pixel
  float u = p_init[b][n][0]; // initial x (can come from warm start)
  float v = p_init[b][n][1]; // initial y (can come from warm start)

  // Keep pixels in [1, w-2] and [1, h-2] so bilinear access to (u+1,v+1) is safe.
  clamp(u, 1, w-2);
  clamp(v, 1, h-2);

  // Setup rays and gradients
  float r[3];
  float gx[3];
  float gy[3];
  float err[3];

  float lambda = lambda_init; // LM damping
  for (int i=0; i<max_iter; i++) {
    // Bilinear interpolation at current pixel (u,v).
    // We interpolate:
    //   r  : ray direction
    //   gx : dr/du (precomputed image gradient in x)
    //   gy : dr/dv (precomputed image gradient in y)
    int u11 = static_cast<int>(floor(u)); // left/top integer x
    int v11 = static_cast<int>(floor(v)); // left/top integer y
    float du = u - static_cast<float>(u11); // x fractional part
    float dv = v - static_cast<float>(v11); // y fractional part

    // Clamping always ensures full bilinear is fine to calculate
    float w11 = du * dv;             // bilinear weight for (u11+1, v11+1)
    float w12 = (1.0-du) * dv;       // bilinear weight for (u11,   v11+1)
    float w21 = du * (1.0-dv);       // bilinear weight for (u11+1, v11)
    float w22 = (1.0-du) * (1.0-dv); // bilinear weight for (u11,   v11)

    // NOTE: Pixels are opposite the area calc!
    float const* r11 = &rays_img[b][v11+1][u11+1][0]; // ptr to sample 11
    float const* r12 = &rays_img[b][v11+1][u11][0];   // ptr to sample 12
    float const* r21 = &rays_img[b][v11][u11+1][0];   // ptr to sample 21
    float const* r22 = &rays_img[b][v11][u11][0];     // ptr to sample 22

    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j]; // interpolated ray
    }
    #pragma unroll
    for (int j=3; j<6; j++) {
      gx[j-3] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j]; // interpolated d(ray)/du
    }
    #pragma unroll
    for (int j=6; j<9; j++) {
      gy[j-6] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j]; // interpolated d(ray)/dv
    }

    // Normalize ray
    float r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]); // norm of ray vector
    float r_norm_inv = 1.0/r_norm;                            // reciprocal for normalization
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv; // normalize ray
    }

    // Calculate error
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j]; // residual in ray space
    }
    float cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2]; // old objective

    // Build tiny LM normal equation:
    //   min || r(u,v) - x_hat ||^2
    // with Jacobian J = [gx, gy] in R^{3x2}.
    //
    // Then:
    //   A = J^T J + lambda * I   (2x2)
    //   b = -J^T e               (2x1)
    //   delta = A^{-1} b
    //
    // A entries:
    //   A00 = gx·gx, A01 = gx·gy, A11 = gy·gy
    float A00 = gx[0]*gx[0] + gx[1]*gx[1] + gx[2]*gx[2];
    float A01 = gx[0]*gy[0] + gx[1]*gy[1] + gx[2]*gy[2];
    float A11 = gy[0]*gy[0] + gy[1]*gy[1] + gy[2]*gy[2];
    // - J^T r
    float b0 = - (err[0]*gx[0] + err[1]*gx[1] + err[2]*gx[2]);
    float b1 = - (err[0]*gy[0] + err[1]*gy[1] + err[2]*gy[2]);
    // LM diagonal
    A00 += lambda; // LM damping on diagonal
    A11 += lambda; // LM damping on diagonal

    // Solve 2x2 linear system in closed form.
    float det_inv = 1.0/(A00*A11 - A01*A01);    // inverse determinant
    float delta_u = det_inv * ( A11*b0 - A01*b1); // update in u
    float delta_v = det_inv * (-A01*b0 + A00*b1); // update in v

    // Get new pixel
    float u_new = u + delta_u; // candidate x after one LM step
    float v_new = v + delta_v; // candidate y after one LM step
    clamp(u_new, 1, w-2);
    clamp(v_new, 1, h-2);


    // Evaluate candidate step.
    u11 = static_cast<int>(floor(u_new)); // integer x at candidate
    v11 = static_cast<int>(floor(v_new)); // integer y at candidate
    du = u_new - u11;                     // frac x at candidate
    dv = v_new - v11;                     // frac y at candidate

    w11 = du * dv; // top left
    w12 = (1.0-du) * dv; // top right
    w21 = du * (1.0-dv); // bottom left
    w22 = (1.0-du) * (1.0-dv); // bottom right

    // NOTE: Pixels are opposite the area calc!
    r11 = &rays_img[b][v11+1][u11+1][0]; // bottom right
    r12 = &rays_img[b][v11+1][u11][0]; // bottom left
    r21 = &rays_img[b][v11][u11+1][0]; // top right
    r22 = &rays_img[b][v11][u11][0]; // top left

    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j]; // candidate ray interpolation
    }
    r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r_norm_inv = 1.0/r_norm;
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv; // normalize candidate ray
    }
    // Calculate error
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j]; // candidate residual
    }
    float new_cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2]; // candidate objective

    // Levenberg-Marquardt trust update:
    // - If step reduces cost: accept step, decrease lambda (more GN-like)
    // - Else: reject step, increase lambda (more gradient-descent-like)
    if (new_cost < cost) {
      u = u_new;                         // accept x
      v = v_new;                         // accept y
      lambda *= 0.1;                     // trust more GN
      converged[b][n] = new_cost < cost_thresh; // convergence flag
    }
    else {
      lambda *= 10.0;               // trust less GN, more damping
      converged[b][n] = cost < cost_thresh; // keep previous convergence state from old cost
    }

  }

  p_new[b][n][0] = u; // final optimized x
  p_new[b][n][1] = v; // final optimized y

}



std::vector<torch::Tensor> iter_proj_cuda(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh)
{
  // Host wrapper for iterative projection kernel.
  const auto batch_size = p_init.size(0);
  const auto n = p_init.size(1);

  const dim3 blocks((n + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK);

  auto opts = p_init.options();
  torch::Tensor p_new = torch::zeros(
    {batch_size, n, 2}, opts);

  auto opts_bool = opts.dtype(torch::kBool);
  torch::Tensor converged = torch::zeros(
    {batch_size, n}, opts_bool);

  iter_proj_kernel<<<blocks, threads>>>(
    rays_img_with_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    pts_3d_norm.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    p_init.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    p_new.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    converged.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
    max_iter,
    lambda_init,
    cost_thresh
  );

  return {p_new, converged};

}
