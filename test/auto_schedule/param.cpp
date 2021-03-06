#include "Halide.h"
#include <stdio.h>

using namespace Halide;

void run_test_1() {
    Param<int> offset;
    ImageParam input(UInt(8), 2);
    Var x("x"), y("y");
    Func f("f");
    f(x, y) = input(x, y) * 2;

    Func g("g");
    g(x, y) = f(x + offset, y) + f(x - offset, y);

    g.estimate(x, 0, 1000).estimate(y, 0, 1000);

    // Auto schedule the pipeline
    Target target = get_target_from_environment();
    Pipeline p(g);

    p.auto_schedule(target);

    // Inspect the schedule
    g.print_loop_nest();
}

void run_test_2() {
    Param<int> offset;
    offset.estimate(1);
    ImageParam input(UInt(8), 2);
    Var x("x"), y("y");
    Func f("f");
    f(x, y) = input(x, y) * 2;

    Func g("g");
    g(x, y) = f(x + offset, y) + f(x - offset, y);

    g.estimate(x, 0, 1000).estimate(y, 0, 1000);

    // Auto schedule the pipeline
    Target target = get_target_from_environment();
    Pipeline p(g);

    p.auto_schedule(target);

    // Inspect the schedule
    g.print_loop_nest();
}

int main(int argc, char **argv) {
    std::cout << "Test 1:" << std::endl;
    run_test_1();
    std::cout << "Test 2:" << std::endl;
    run_test_2();
    printf("Success!\n");
    return 0;
}
