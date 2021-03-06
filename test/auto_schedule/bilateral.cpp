#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_image.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;

double run_test(bool auto_schedule) {    
    Image<float> input = load_and_convert_image("../../../images/gray.png");

    float r_sigma = 0.1;
    int s_sigma = 8;
    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma/2, y * s_sigma + r.y - s_sigma/2);
    val = clamp(val, 0.0f, 1.0f);
    Expr zi = cast<int>(val * (1.0f/r_sigma) + 0.5f);
    Func histogram("histogram");
    histogram(x, y, z, c) = 0.0f;
    histogram(x, y, zi, c) += select(c == 0, val, 1.0f);

    // TODO: compute the estimate from the parameter values
    histogram.estimate(z, -2, 16);

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurz(x, y, z, c) = (histogram(x, y, z-2, c) +
                         histogram(x, y, z-1, c)*4 +
                         histogram(x, y, z  , c)*6 +
                         histogram(x, y, z+1, c)*4 +
                         histogram(x, y, z+2, c));
    blurx(x, y, z, c) = (blurz(x-2, y, z, c) +
                         blurz(x-1, y, z, c)*4 +
                         blurz(x  , y, z, c)*6 +
                         blurz(x+1, y, z, c)*4 +
                         blurz(x+2, y, z, c));
    blury(x, y, z, c) = (blurx(x, y-2, z, c) +
                         blurx(x, y-1, z, c)*4 +
                         blurx(x, y  , z, c)*6 +
                         blurx(x, y+1, z, c)*4 +
                         blurx(x, y+2, z, c));

    // TODO: compute the estimate from the parameter values
    blurz.estimate(z, 0, 13);
    blurx.estimate(z, 0, 13);
    blury.estimate(z, 0, 13);

    // Take trilinear samples to compute the output
    val = clamp(input(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f/r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x/s_sigma;
    Expr yi = y/s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y, c) =
        lerp(lerp(lerp(blury(xi, yi, zi, c), blury(xi+1, yi, zi, c), xf),
                  lerp(blury(xi, yi+1, zi, c), blury(xi+1, yi+1, zi, c), xf), yf),
             lerp(lerp(blury(xi, yi, zi+1, c), blury(xi+1, yi, zi+1, c), xf),
                  lerp(blury(xi, yi+1, zi+1, c), blury(xi+1, yi+1, zi+1, c), xf), yf), zf);

    // Normalize
    Func bilateral_grid("bilateral_grid");
    bilateral_grid(x, y) = interpolated(x, y, 0)/interpolated(x, y, 1);
    bilateral_grid.estimate(x, 0, input.width ()).estimate(y, 0, input.height());

    Target target = get_target_from_environment();
    Pipeline p(bilateral_grid);

    if (!auto_schedule) {
        if (target.has_gpu_feature()) {
            /*// Schedule blurz in 8x8 tiles. This is a tile in
            // grid-space, which means it represents something like
            // 64x64 pixels in the input (if s_sigma is 8).
            blurz.compute_root().reorder(c, z, x, y).gpu_tile(x, y, 8, 8);

            // Schedule histogram to happen per-tile of blurz, with
            // intermediate results in shared memory. This means histogram
            // and blurz makes a three-stage kernel:
            // 1) Zero out the 8x8 set of histograms
            // 2) Compute those histogram by iterating over lots of the input image
            // 3) Blur the set of histograms in z
            histogram.reorder(c, z, x, y).compute_at(blurz, Var::gpu_blocks()).gpu_threads(x, y);
            histogram.update().reorder(c, r.x, r.y, x, y).gpu_threads(x, y).unroll(c);

            // An alternative schedule for histogram that doesn't use shared memory:
            // histogram.compute_root().reorder(c, z, x, y).gpu_tile(x, y, 8, 8);
            // histogram.update().reorder(c, r.x, r.y, x, y).gpu_tile(x, y, 8, 8).unroll(c);

            // Schedule the remaining blurs and the sampling at the end similarly.
            blurx.compute_root().gpu_tile(x, y, z, 8, 8, 1);
            blury.compute_root().gpu_tile(x, y, z, 8, 8, 1);
            bilateral_grid.compute_root().gpu_tile(x, y, s_sigma, s_sigma);*/
        } else {
            // The CPU schedule.
            blurz.compute_root().reorder(c, z, x, y).parallel(y).vectorize(x, 8).unroll(c);
            //histogram.compute_at(blurz, y);
            //histogram.update().reorder(c, r.x, r.y, x, y).unroll(c);
            histogram.compute_root().reorder(c, y).parallel(y);
            histogram.update().reorder(c, y).parallel(y);
            blurx.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 8).unroll(c);
            blury.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 8).unroll(c);
            bilateral_grid.compute_root().parallel(y).vectorize(x, 8);
        }
    } else {
#ifndef CPU
       p.auto_schedule(target);
#else
    #ifndef PARALLELISM
        #error "PARALLELISM Not Set"
    #endif
    #ifndef L2_CACHE_SIZE
        #error "L2_CACHE_SIZE Not Set"
    #endif    
       MachineParams arch_params (PARALLELISM, L2_CACHE_SIZE, 40);
       p.auto_schedule(target, arch_params);
#endif   
    }

    std::cout << "Schedule:"<< std::endl;
    // Inspect the schedule
    bilateral_grid.print_loop_nest();

    // Benchmark the schedule
    Buffer<float> out(input.width(), input.height());
    double t = benchmark(5, 10, [&]() {
        p.realize(out);
    });

    return t*1000;
}

int main(int argc, char **argv) {
    double manual_time = run_test(false);
    double auto_time = run_test(true);

    std::cout << "======================" << std::endl;
    std::cout << "Manual time: " << manual_time << "ms" << std::endl;
    std::cout << "Auto time: " << auto_time << "ms" << std::endl;
    std::cout << "======================" << std::endl;
    printf("Success!\n");
    return 0;
}
