#include "Halide.h"
#include "halide_benchmark.h"
#include <stdio.h>
#include "halide_image.h"
#include "halide_image_io.h"
using namespace Halide;
using namespace Halide::Tools;

Expr sum3x3(Func f, Var x, Var y) {
    return f(x-1, y-1) + f(x-1, y) + f(x-1, y+1) +
            f(x, y-1)   + f(x, y)   + f(x, y+1) +
            f(x+1, y-1) + f(x+1, y) + f(x+1, y+1);
}

double run_test(bool auto_schedule) {
    
    /*for (int y = 0; y < in.height(); y++) {
        for (int x = 0; x < in.width(); x++) {
            for (int c = 0; c < 3; c++) {
                in(x, y, c) = rand() & 0xfff;
            }
        }
    }*/

    Buffer<float> in = load_and_convert_image("../../../images/venice_wikimedia.jpg");

    Func in_b = BoundaryConditions::repeat_edge(in);

    Var x, y, c;

    Func gray("gray");
    gray(x, y) = 0.299f * in_b(x, y, 0) + 0.587f * in_b(x, y, 1) + 0.114f * in_b(x, y, 2);


    Func Iy("Iy");
    Iy(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x-1, y+1)*(1.0f/12) +
            gray(x, y-1)*(-2.0f/12) + gray(x, y+1)*(2.0f/12) +
            gray(x+1, y-1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);


    Func Ix("Ix");
    Ix(x, y) = gray(x-1, y-1)*(-1.0f/12) + gray(x+1, y-1)*(1.0f/12) +
            gray(x-1, y)*(-2.0f/12) + gray(x+1, y)*(2.0f/12) +
            gray(x-1, y+1)*(-1.0f/12) + gray(x+1, y+1)*(1.0f/12);


    Func Ixx("Ixx");
    Ixx(x, y) = Ix(x, y) * Ix(x, y);

    Func Iyy("Iyy");
    Iyy(x, y) = Iy(x, y) * Iy(x, y);

    Func Ixy("Ixy");
    Ixy(x, y) = Ix(x, y) * Iy(x, y);

    Func Sxx("Sxx");

    Sxx(x, y) = sum3x3(Ixx, x, y);

    Func Syy("Syy");
    Syy(x, y) = sum3x3(Iyy, x, y);


    Func Sxy("Sxy");
    Sxy(x, y) = sum3x3(Ixy, x, y);

    Func det("det");
    det(x, y) = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);

    Func trace("trace");
    trace(x, y) = Sxx(x, y) + Syy(x, y);

    Func harris("harris");
    harris(x, y) = det(x, y) - 0.04f * trace(x, y) * trace(x, y);

    Func shifted("shifted");
    shifted(x, y) = harris(x + 2, y + 2);

    shifted.estimate(x, 0, in.width()).estimate(y, 0, in.height());

    Target target = get_target_from_environment();
    Pipeline p(shifted);

    if (!auto_schedule) {
        Var yi, xi;
        if (target.has_gpu_feature()) {
            shifted.gpu_tile(x, y, xi, yi, 14, 14);
            Ix.compute_at(shifted, x).gpu_threads(x, y);
            Iy.compute_at(shifted, x).gpu_threads(x, y);
        } else {
//Original Schedule:
            shifted.tile(x, y, xi, yi, 128, 128)
                   .vectorize(xi, 8).parallel(y);
            Ix.compute_at(shifted, x).vectorize(x, 8);
            Iy.compute_at(shifted, x).vectorize(x, 8);
            
//PolyMage Schedule
/*   shifted.reorder(y,x).tile(x, y, xi, yi, 256, 23)
                    .vectorize(xi, 8).parallel(y);//.parallel(x);
           Ix.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            Iy.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            det.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            trace.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            Sxx.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            Syy.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            Ixy.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);        
            Ixx.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);        
            Iyy.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
            gray.store_at(shifted, x).compute_at(shifted, yi).vectorize(x, 8);
*/
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

    // Inspect the schedule
    shifted.print_loop_nest();

    // Run the schedule
    Buffer<float> out(in.width(), in.height());
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
