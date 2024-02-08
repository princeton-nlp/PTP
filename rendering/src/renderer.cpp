#include "render.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <vector>
#include <tuple>

// Interface

namespace py = pybind11;

py::object render(py::array_t<unsigned char> &array, std::string text,
    int height, int width, int fontsize, int line_space, bool fixed_width, 
    bool fix_spacing, bool no_partial) {
    auto buf = array.request();
    
    unsigned char *ptr = (unsigned char *) buf.ptr;
    
    std::string rendered = render_text_onto_array(text, ptr, height, 
        width, fontsize, line_space, fixed_width, fix_spacing, no_partial);

    return make_tuple(array, rendered);
}

py::object render_unicode(py::array_t<unsigned char> &array, 
    std::string text, int height, int width, int fontsize, 
    int line_space, bool fixed_width, bool fix_spacing, bool no_partial, bool no_margin, bool fixed_offset) {
    auto buf = array.request();
    
    unsigned char *ptr = (unsigned char *) buf.ptr;
    
    std::string rendered = render_text_onto_array_unicode(text, ptr, height,
        width, fontsize, line_space, fixed_width, fix_spacing, no_partial, no_margin, fixed_offset);

    return make_tuple(array, rendered);
}

py::object only_render_unicode(py::array_t<unsigned char> &array, 
    std::string text, int height, int width, int fontsize, 
    int line_space, bool fixed_width, bool fix_spacing, bool no_partial) {
    auto buf = array.request();
    
    unsigned char *ptr = (unsigned char *) buf.ptr;
    
    only_render_text_onto_array_unicode(text, ptr, height,
        width, fontsize, line_space, fixed_width, fix_spacing, no_partial);

    return array;
}

PYBIND11_MODULE(renderer, m) {
    m.doc() = "pybind11 plugin for rendering text"; 

    m.def("render", &render, "A function that renders text");
    m.def("render_unicode", &render_unicode, 
        "A function that renders unicode text");
    m.def("only_render_unicode", &only_render_unicode, 
        "A function that renders unicode text");
}