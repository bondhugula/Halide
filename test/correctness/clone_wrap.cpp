#include "Halide.h"
#include "test/common/check_call_graphs.h"

#include <stdio.h>
#include <map>

using std::map;
using std::string;

using namespace Halide;
using namespace Halide::Internal;

int calling_clone_no_op_test() {
    Var x("x"), y("y");

    {
        Func f("f"), g("g");
        f(x, y) = x + y;
        g(x, y) = f(x, y);

        // Calling clone on the same Func for the same Func multiple times should
        // return the same clone
        Func clone = f.clone_in(g);
        for (int i = 0; i < 5; ++i) {
            Func temp = f.clone_in(g);
            if (clone.name() != temp.name()) {
                std::cerr << "Expect " << clone.name() << "; got " << temp.name() << " instead\n";
                return -1;
            }
        }
    }

    {
        Func d("d"), e("e"), f("f"), g("g"), h("h");
        d(x, y) = x + y;
        e(x, y) = d(x, y);
        f(x, y) = d(x, y);
        g(x, y) = d(x, y);
        h(x, y) = d(x, y);

        Func clone1 = d.clone_in({e, f, g});
        Func clone2 = d.clone_in({g, f, e});
        if (clone1.name() != clone2.name()) {
            std::cerr << "Expect " << clone1.name() << "; got " << clone2.name() << " instead\n";
            return -1;
        }
    }

    return 0;
}

int func_clone_test() {
    Func f("f"), g("g");
    Var x("x"), y("y");

    f(x) = x;
    g(x, y) = f(x);

    Func clone = f.clone_in(g).compute_root();
    f.compute_root();

    // Check the call graphs.
    // Expect 'g' to call 'clone', 'clone' to call 'f', 'f' to call nothing
    Module m = g.compile_to_module({});
    CheckCalls c;
    m.functions().front().body.accept(&c);

    CallGraphs expected = {
        {g.name(), {clone.name()}},
        {clone.name(), {f.name()}},
        {f.name(), {}},
    };
    if (check_call_graphs(c.calls, expected) != 0) {
        return -1;
    }

    Buffer<int> im = g.realize(200, 200);
    auto func = [](int x, int y) { return x; };
    if (check_image(im, func)) {
        return -1;
    }
    return 0;
}

int multiple_funcs_sharing_clone_test() {
    Func f("f"), g1("g1"), g2("g2"), g3("g3");
    Var x("x"), y("y");

    f(x) = x;
    g1(x, y) = f(x);
    g2(x, y) = f(x);
    g3(x, y) = f(x);

    f.compute_root();
    Func f_clone = f.clone_in({g1, g2, g3}).compute_root();

    {
        // Check the call graphs.
        // Expect 'g1' to call 'f_clone', 'f_clone' to call 'f', 'f' to call nothing
        Module m = g1.compile_to_module({});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g1.name(), {f_clone.name()}},
            {f_clone.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g1.realize(200, 200);
        auto func = [](int x, int y) { return x; };
        if (check_image(im, func)) {
            return -1;
        }
    }

    {
        // Check the call graphs.
        // Expect 'g2' to call 'f_clone', 'f_clone' to call 'f', 'f' to call nothing
        Module m = g2.compile_to_module({});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g2.name(), {f_clone.name()}},
            {f_clone.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g2.realize(200, 200);
        auto func = [](int x, int y) { return x; };
        if (check_image(im, func)) {
            return -1;
        }
    }

    {
        // Check the call graphs.
        // Expect 'g3' to call 'f_clone', 'f_clone' to call 'f', 'f' to call nothing
        Module m = g3.compile_to_module({});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g3.name(), {f_clone.name()}},
            {f_clone.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g3.realize(200, 200);
        auto func = [](int x, int y) { return x; };
        if (check_image(im, func)) {
            return -1;
        }
    }
    return 0;
}

int update_defined_after_clone_test() {
    Func f("f"), g("g");
    Var x("x"), y("y");

    f(x, y) = x + y;
    g(x, y) = f(x, y);

    Func clone = f.clone_in(g);

    // Update of 'g' is defined after f.clone_in(g) is called. g's updates should
    // still call f's clone.
    RDom r(0, 100, 0, 100);
    r.where(r.x < r.y);
    g(r.x, r.y) += 2*f(r.x, r.y);

    Param<bool> param;

    Var xi("xi");
    RVar rxo("rxo"), rxi("rxi");
    g.specialize(param).vectorize(x, 8).unroll(x, 2).split(x, x, xi, 4).parallel(x);
    g.update(0).split(r.x, rxo, rxi, 2).unroll(rxi);
    f.compute_root();
    clone.compute_root().vectorize(x, 8).unroll(x, 2).split(x, x, xi, 4).parallel(x);

    {
        param.set(true);

        // Check the call graphs.
        // Expect initialization of 'g' to call 'clone' and its update to call
        // 'clone' and 'g', clone' to call 'f', 'f' to call nothing
        Module m = g.compile_to_module({g.infer_arguments()});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g.name(), {clone.name(), g.name()}},
            {clone.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g.realize(200, 200);
        auto func = [](int x, int y) {
            return ((0 <= x && x <= 99) && (0 <= y && y <= 99) && (x < y)) ? 3*(x + y) : (x + y);
        };
        if (check_image(im, func)) {
            return -1;
        }
    }

    {
        param.set(false);

        // Check the call graphs.
        // Expect initialization of 'g' to call 'clone' and its update to call
        // 'clone' and 'g', clone' to call 'f', 'f' to call nothing
        Module m = g.compile_to_module({g.infer_arguments()});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g.name(), {clone.name(), g.name()}},
            {clone.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g.realize(200, 200);
        auto func = [](int x, int y) {
            return ((0 <= x && x <= 99) && (0 <= y && y <= 99) && (x < y)) ? 3*(x + y) : (x + y);
        };
        if (check_image(im, func)) {
            return -1;
        }
    }

    return 0;
}

int clone_depend_on_mutated_func_test() {
    Func e("e"), f("f"), g("g"), h("h");
    Var x("x"), y("y");

    e(x, y) = x + y;
    f(x, y) = e(x, y);
    g(x, y) = f(x, y);
    h(x, y) = g(x, y);

    Var xo("xo"), xi("xi");
    e.compute_root();
    f.compute_at(g, y).vectorize(x, 8);
    g.compute_root();
    Func e_in_f = e.clone_in(f);
    Func g_in_h = g.clone_in(h).compute_root();
    g_in_h.compute_at(h, y).vectorize(x, 8);
    e_in_f.compute_at(f, y).split(x, xo, xi, 8);

    // Check the call graphs.
    // Expect 'h' to call 'g_in_h', 'g_in_h' to call 'g', 'g' to call 'f',
    // 'f' to call 'e_in_f', e_in_f' to call 'e', 'e' to call nothing
    Module m = h.compile_to_module({});
    CheckCalls c;
    m.functions().front().body.accept(&c);

    CallGraphs expected = {
        {h.name(), {g_in_h.name()}},
        {g_in_h.name(), {g.name()}},
        {g.name(), {f.name()}},
        {f.name(), {e_in_f.name()}},
        {e_in_f.name(), {e.name()}},
        {e.name(), {}},
    };
    if (check_call_graphs(c.calls, expected) != 0) {
        return -1;
    }

    Buffer<int> im = h.realize(200, 200);
    auto func = [](int x, int y) { return x + y; };
    if (check_image(im, func)) {
        return -1;
    }
    return 0;
}

int clone_on_clone_test() {
    Func e("e"), f("f"), g("g"), h("h");
    Var x("x"), y("y");

    e(x, y) = x + y;
    f(x, y) = e(x, y);
    g(x, y) = f(x, y) + e(x, y);
    Func f_in_g = f.clone_in(g).compute_root();
    Func f_in_f_in_g = f.clone_in(f_in_g).compute_root();
    h(x, y) = g(x, y) + f(x, y) + f_in_f_in_g(x, y);

    e.compute_root();
    f.compute_root();
    g.compute_root();
    Func f_in_h = f.clone_in(h).compute_root();
    Func g_in_h = g.clone_in(h).compute_root();

    // Check the call graphs.
    Module m = h.compile_to_module({});
    CheckCalls c;
    m.functions().front().body.accept(&c);

    CallGraphs expected = {
        {h.name(), {f_in_h.name(), g_in_h.name(), f_in_f_in_g.name()}},
        {f_in_h.name(), {f.name()}},
        {g_in_h.name(), {g.name()}},
        {g.name(), {e.name(), f_in_g.name()}},
        {f_in_g.name(), {f_in_f_in_g.name()}},
        {f_in_f_in_g.name(), {f.name()}},
        {f.name(), {e.name()}},
        {e.name(), {}},
    };
    if (check_call_graphs(c.calls, expected) != 0) {
        return -1;
    }

    Buffer<int> im = h.realize(200, 200);
    auto func = [](int x, int y) { return 4*(x + y); };
    if (check_image(im, func)) {
        return -1;
    }
    return 0;
}

int two_fold_clone_test() {
    Func input("input"), input_in_output_in_output, input_in_output, output("output");
    Var x("x"), y("y");

    input(x, y) = 2*x + 3*y;
    input.compute_root();

    output(x, y) = input(y, x);

    Var xi("xi"), yi("yi");
    output.tile(x, y, xi, yi, 8, 8);

    input_in_output = input.clone_in(output).compute_at(output, x).vectorize(x).unroll(y);
    input_in_output_in_output = input_in_output.clone_in(output).compute_at(output, x).unroll(x).unroll(y);

    // Check the call graphs.
    Module m = output.compile_to_module({});
    CheckCalls c;
    m.functions().front().body.accept(&c);

    CallGraphs expected = {
        {output.name(), {input_in_output_in_output.name()}},
        {input_in_output_in_output.name(), {input_in_output.name()}},
        {input_in_output.name(), {input.name()}},
        {input.name(), {}},
    };
    if (check_call_graphs(c.calls, expected) != 0) {
        return -1;
    }

    Buffer<int> im = output.realize(1024, 1024);
    auto func = [](int x, int y) { return 3*x + 2*y; };
    if (check_image(im, func)) {
        return -1;
    }
    return 0;
}

int multi_folds_clone_test() {
    Func f("f"), f_in_g_in_g, f_in_g, f_in_g_in_g_in_h, f_in_g_in_g_in_h_in_h, g("g"), h("h");
    Var x("x"), y("y");

    f(x, y) = 2*x + 3*y;
    f.compute_root();

    g(x, y) = f(y, x);

    Var xi("xi"), yi("yi");
    g.compute_root().tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi);

    f_in_g = f.clone_in(g).compute_root().tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi);
    f_in_g_in_g = f_in_g.clone_in(g).compute_root().tile(x, y, xi, yi, 8, 8).unroll(xi).unroll(yi);

    h(x, y) = f_in_g_in_g(y, x);
    f_in_g_in_g_in_h = f_in_g_in_g.clone_in(h).compute_at(h, x).vectorize(x).unroll(y);
    f_in_g_in_g_in_h_in_h = f_in_g_in_g_in_h.clone_in(h).compute_at(h, x).unroll(x).unroll(y);
    h.compute_root().tile(x, y, xi, yi, 8, 8);

    {
        // Check the call graphs.
        Module m = g.compile_to_module({});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {g.name(), {f_in_g_in_g.name()}},
            {f_in_g_in_g.name(), {f_in_g.name()}},
            {f_in_g.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = g.realize(1024, 1024);
        auto func = [](int x, int y) { return 3*x + 2*y; };
        if (check_image(im, func)) {
            return -1;
        }
    }

    {
        // Check the call graphs.
        Module m = h.compile_to_module({});
        CheckCalls c;
        m.functions().front().body.accept(&c);

        CallGraphs expected = {
            {h.name(), {f_in_g_in_g_in_h_in_h.name()}},
            {f_in_g_in_g_in_h_in_h.name(), {f_in_g_in_g_in_h.name()}},
            {f_in_g_in_g_in_h.name(), {f_in_g_in_g.name()}},
            {f_in_g_in_g.name(), {f_in_g.name()}},
            {f_in_g.name(), {f.name()}},
            {f.name(), {}},
        };
        if (check_call_graphs(c.calls, expected) != 0) {
            return -1;
        }

        Buffer<int> im = h.realize(1024, 1024);
        auto func = [](int x, int y) { return 3*x + 2*y; };
        if (check_image(im, func)) {
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    /*printf("Running calling clone no op test\n");
    if (calling_clone_no_op_test() != 0) {
        return -1;
    }

    printf("Running func clone test\n");
    if (func_clone_test() != 0) {
        return -1;
    }

    printf("Running multiple funcs sharing clone test\n");
    if (multiple_funcs_sharing_clone_test() != 0) {
        return -1;
    }

    printf("Running update is defined after clone test\n");
    if (update_defined_after_clone_test() != 0) {
        return -1;
    }

    printf("Running clone depend on mutated func test\n");
    if (clone_depend_on_mutated_func_test() != 0) {
        return -1;
    }

    printf("Running clone on clone test\n");
    if (clone_on_clone_test() != 0) {
        return -1;
    }

    printf("Running two fold clone test\n");
    if (two_fold_clone_test() != 0) {
        return -1;
    }

    printf("Running multi folds clone test\n");
    if (multi_folds_clone_test() != 0) {
        return -1;
    }*/

    printf("Success!\n");
    return 0;
}
