#ifndef RENDER_H_
#define RENDER_H_

#include <iostream>
#include <string>
#include <math.h>
#include <iconv.h>
#include <climits>

#include <ft2build.h>
#include FT_FREETYPE_H

void init_ft(FT_Library &library);
void init_face(FT_Library library, FT_Face &face, std::string 
    fontpath="GoNotoCurrent.ttf", unsigned int fontsize=16);
void show_image_in_terminal(unsigned char *image, unsigned int height=256, 
    unsigned int width=512);
void finish(FT_Library &library, FT_Face &face);

std::string render_text(FT_Face &face, std::string text, unsigned char *array, 
    unsigned int height=256, unsigned int width=512, 
    unsigned int fontsize=16, int line_space=-1, bool fixed_width=true, 
    bool fix_spacing=true, bool no_partial=false);
std::string render_text_unicode(FT_Face &face, std::string text, 
    unsigned char *array, unsigned int height=256, unsigned int width=512, 
    unsigned int fontsize=16, int line_space=-1, bool fixed_width=true, 
    bool fix_spacing=true, bool no_partial=false, bool no_margin=false, bool fixed_offset=false);
std::string render_text_onto_array(std::string text, unsigned char *array, 
    int height=256, int width=512, int fontsize=16, int line_space=-1,
    bool fixed_width=true, bool fix_spacing=true, bool no_partial=false,
    std::string fontpath="GoNotoCurrent.ttf");
std::string render_text_onto_array_unicode(std::string text, unsigned char *array, 
    int height=256, int width=512, int fontsize=16, int line_space=-1, 
    bool fixed_width=true, bool fix_spacing=true, bool no_partial=false, bool no_margin=false, bool fixed_offset=false,
    std::string fontpath="GoNotoCurrent.ttf");

// These functions do not return the rendered part as a string
void only_render_text_unicode(FT_Face &face, std::string text, unsigned char *array, 
    unsigned int height=256, unsigned int width=512, 
    unsigned int fontsize=16, int line_space=-1, bool fixed_width=true, 
    bool fix_spacing=true, bool no_partial=false);
void only_render_text_onto_array_unicode(std::string text, 
    unsigned char *array, int height=256, int width=512, int fontsize=16, 
    int line_space=-1, bool fixed_width=true, bool fix_spacing=true, 
    bool no_partial=false, std::string fontpath="GoNotoCurrent.ttf");

#endif