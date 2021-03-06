#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {

    int H = 6400;
    int W = 4800;
    Buffer<uint16_t> input(H, W);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

    Var x("x"), y("y");

    int num_stencils = 15;

    std::vector<Func> stencils;
    for(int i = 0; i < num_stencils; i ++) {
        Func s("stencil_" + std::to_string(i));
        stencils.push_back(s);
    }

    stencils[0](x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    for (int i = 1; i < num_stencils; i++) {
        stencils[i](x, y) = (stencils[i-1](x, y) + stencils[i-1](x, y+1) +
                             stencils[i-1](x, y+2))/3;
    }

    // Specifying estimates
    stencils[num_stencils - 1].estimate(x, 0, 6200).estimate(y, 0, 4600);

    // Auto schedule the pipeline
    Target target = get_target_from_environment();
    Pipeline p(stencils[num_stencils - 1]);

    p.auto_schedule(target);

    // Inspect the schedule
    stencils[num_stencils - 1].print_loop_nest();

    // Run the schedule
    Buffer<uint16_t> out = p.realize(6204, 4604);

    printf("Success!\n");
    return 0;
}
