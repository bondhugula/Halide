#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "halide_image.h"


using namespace Halide::Tools;

using namespace Halide;

Var x, y, c;

// Downsample with a [1 4 6 4 1] filter
Func downsample(Func f) {
    Func downx, downy;

    downx(x, y, _) = (
                       f(2*x-2, y, _) * 1.0f
                     + f(2*x-1, y, _) * 4.0f
                     + f(2*x+0, y, _) * 6.0f
                     + f(2*x+1, y, _) * 4.0f
                     + f(2*x+2, y, _) * 1.0f
                     ) / 16.0f;

    downy(x, y, _) = (
                       downx(x, 2*y-2, _) * 1.0f
                     + downx(x, 2*y-1, _) * 4.0f
                     + downx(x, 2*y+0, _) * 6.0f
                     + downx(x, 2*y+1, _) * 4.0f
                     + downx(x, 2*y+2, _) * 1.0f
                     ) / 16.0f;

    return downy;
}

// Upsample using a [1 4 6 4 1]' filter
Func upsample(Func f) {
    Func upx, upy;

    upx(x, y, _) = select( x%2 == 1,
                                 (
                                  f((x-1)/2, y, _) * 4.0f +
                                  f((x+1)/2, y, _) * 4.0f
                                 ) / 8.0f // for odd x
                                ,(
                                  f(x/2-1, y, _) * 1.0f +
                                  f(x/2+0, y, _) * 6.0f +
                                  f(x/2+1, y, _) * 1.0f
                                 ) / 8.0f); // for even x

    upy(x, y, _) = select( y%2 == 1,
                                 (
                                  upx(x, (y-1)/2, _) * 4.0f +
                                  upx(x, (y+1)/2, _) * 4.0f
                                 ) / 8.0f // for odd y
                                ,(
                                  upx(x, y/2-1, _) * 1.0f +
                                  upx(x, y/2+0, _) * 6.0f +
                                  upx(x, y/2+1, _) * 1.0f
                                 ) / 8.0f); // for even y

    return upy;
}

Func lapl(Func fd, Func f) {
    Func upx, upy;

    upx(x, y, _) = select( x%2 == 1,
                                 (
                                  fd((x-1)/2, y, _) * 4.0f +
                                  fd((x+1)/2, y, _) * 4.0f
                                 ) / 8.0f // for odd x
                                ,(
                                  fd(x/2-1, y, _) * 1.0f +
                                  fd(x/2+0, y, _) * 6.0f +
                                  fd(x/2+1, y, _) * 1.0f
                                 ) / 8.0f); // for even x

    upy(x, y, _) = select( y%2 == 1,
                                 f(x, y, _) - (
                                  upx(x, (y-1)/2, _) * 4.0f +
                                  upx(x, (y+1)/2, _) * 4.0f
                                 ) / 8.0f // for odd y
                                ,f(x, y, _) -  (
                                  upx(x, y/2-1, _) * 1.0f +
                                  upx(x, y/2+0, _) * 6.0f +
                                  upx(x, y/2+1, _) * 1.0f
                                 ) / 8.0f); // for even y

    return upy;
}

/*
// Downsample with a 1 3 3 1 filter
Func downsample(Func f) {
    Func downx, downy;

    downx(x, y, _) = (f(2*x-1, y, _) + 3.0f * (f(2*x, y, _) + f(2*x+1, y, _)) + f(2*x+2, y, _)) / 8.0f;
    downy(x, y, _) = (downx(x, 2*y-1, _) + 3.0f * (downx(x, 2*y, _) + downx(x, 2*y+1, _)) + downx(x, 2*y+2, _)) / 8.0f;

    return downy;
}

// Upsample using bilinear interpolation
Func upsample(Func f) {
    Func upx, upy;

    upx(x, y, _) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, _) + 0.75f * f(x/2, y, _);
    upy(x, y, _) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), _) + 0.75f * upx(x, y/2, _);

    return upy;

}
*/

double run_test (bool autosched)
{
    /* THE ALGORITHM */

    // Number of pyramid levels
    const int J = 4;

    // Takes two float inputs and a float mask
    Image<float> input1 = load_and_convert_image("../../../images/grand_canyon1.jpg");
    Image<float> input2 = load_and_convert_image("../../../images/grand_canyon1.jpg");
    Image<float> mask = load_and_convert_image("../../../images/mask_1024_768.png");
    
    // Set boundary conditions
    Func clamped1;
    Func clamped2;
    Func clamped_;
    clamped1(x, y, c) = input1(clamp(x, 0, mask.width()-1), clamp(y, 0, input1.height()-1), c);
    clamped2(x, y, c) = input2(clamp(x, 0, mask.width()-1), clamp(y, 0, input2.height()-1), c);
    clamped_(x, y)    = mask(clamp(x, 0, mask.width()-1), clamp(y, 0, mask.height()-1));

    // Make the Gaussian pyramid for each
    Func gPyramid1[J];
    Func gPyramid2[J];
    Func gPyramid_[J];
    gPyramid1[0](x, y, c) = clamped1(x, y, c);
    gPyramid2[0](x, y, c) = clamped2(x, y, c);
    gPyramid_[0](x, y)    = clamped_(x, y);
    for (int j = 1; j < J; j++) {
        gPyramid1[j](x, y, c) = downsample(gPyramid1[j-1])(x, y, c);
        gPyramid2[j](x, y, c) = downsample(gPyramid2[j-1])(x, y, c);
        gPyramid_[j](x, y)    = downsample(gPyramid_[j-1])(x, y);
    }

    // Construct the laplacian pyramid of the source images
    Func lPyramid1[J];
    Func lPyramid2[J];
    lPyramid1[J-1](x, y, c) = gPyramid1[J-1](x, y, c);
    lPyramid2[J-1](x, y, c) = gPyramid2[J-1](x, y, c);
    for (int j = J-2; j >= 0; j--) {
        lPyramid1[j](x, y, c) = lapl (gPyramid1[j+1], gPyramid1[j])(x, y, c);
        lPyramid2[j](x, y, c) = lapl (gPyramid2[j+1], gPyramid2[j])(x, y, c);
    }

    // Blend the laplacian pyramid of the source images
    Func outLPyramid[J];
    for (int j = 0; j < J; j++) {
        // Use mask to blend the laplacians
        outLPyramid[j](x, y, c) = ( 1.0f - gPyramid_[j](x, y) ) * lPyramid2[j](x, y, c) +
                                  (    gPyramid_[j](x, y)     ) * lPyramid1[j](x, y, c);
        //outLPyramid[j](x, y, c) = ( 1.0f - gPyramid_[j](x, y) ) * 0.0f +
                                  //(    gPyramid_[j](x, y)     ) * 1.0f;
    }

    // Collapse the pyramid
    Func outGPyramid[J];
    outGPyramid[J-1](x, y, c) = outLPyramid[J-1](x, y, c);
    for (int j = J-2; j >= 0; j--) {
        outGPyramid[j](x, y, c) = lapl(outGPyramid[j+1], outLPyramid[j])(x, y, c);
    }

    //Func output;
    //output(x, y, c) = outGPyramid[0](x, y, c);
    //output(x, y, c) = clamp(outGPyramid[0](x, y, c), 0.0f, 1.0f);
    outGPyramid[0](x, y, c) = clamp(outGPyramid[0](x, y, c), 0.0f, 1.0f);

    outGPyramid[0].estimate (x, 0, mask.width ()).estimate(y, 0, mask.height()).estimate(c, 0, 3);
    Target target = get_target_from_environment();
    Pipeline p(outGPyramid[0]);
    
    if (!autosched)
    {
        Var yi;

        for (int j = 0; j < 4 && j < J; j++) {
            if (j > 0){
                gPyramid1[j].compute_root().parallel(y).vectorize(x, 8);
                gPyramid2[j].compute_root().parallel(y).vectorize(x, 8);
                gPyramid_[j].compute_root().parallel(y).vectorize(x, 8);
            }
            if (j > 0){
                lPyramid1[j].compute_root().parallel(y).vectorize(x, 8);
                lPyramid2[j].compute_root().parallel(y).vectorize(x, 8);
            }
            outLPyramid[j].compute_root().parallel(y).vectorize(x, 8);
            outGPyramid[j].compute_root().parallel(y).vectorize(x, 8);
        }
        for (int j = 4; j < J; j++) {
            gPyramid1[j].compute_root().parallel(y).vectorize(x, 8);
            gPyramid2[j].compute_root().parallel(y).vectorize(x, 8);
            gPyramid_[j].compute_root().parallel(y).vectorize(x, 8);
            lPyramid1[j].compute_root().parallel(y).vectorize(x, 8);
            lPyramid2[j].compute_root().parallel(y).vectorize(x, 8);
            outLPyramid[j].compute_root().parallel(y).vectorize(x, 8);
            outGPyramid[j].compute_root().parallel(y).vectorize(x, 8);
        }
    }
    else
    {
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
    
    outGPyramid[0].print_loop_nest();
    
    Buffer<float> out(mask.width (), mask.height(), 3);
    double t = benchmark(5, 50, [&]() {
        p.realize(out);
    });

    return t*1000;
    
    return 0;
}

int main(int argc, char **argv) {
    std::cout << "Compiling Manual Schedule " << std::endl;
    double manual_time = run_test(false);
    std::cout << "Compiling Auto Schedule " << std::endl;
    double auto_time = run_test(true);

    std::cout << "======================" << std::endl;
    std::cout << "Manual time: " << manual_time << "ms" << std::endl;
    std::cout << "Auto time: " << auto_time << "ms" << std::endl;
    std::cout << "======================" << std::endl;
    printf("Success!\n");
    return 0;
}
