#include "render.h"
#include <unordered_map>
#include <stdio.h>
#include <iostream>

void init_ft(FT_Library &library) {
    int error = FT_Init_FreeType(&library);
    if (error) {
        std::cerr << "Failed to initialize FreeType! (" << error << ")\n";
        exit(-1);
    }
}

void init_face(FT_Library library, FT_Face &face, std::string fontpath, 
    unsigned int fontsize) {
    int error = 0;

    error = FT_New_Face(library, fontpath.c_str(), 0, &face);
    if (error) {
        std::cerr << "Failed to initialize Face! (" << error << ")\n";
        exit(-1);
    }

    error = FT_Set_Char_Size(face, 0, fontsize*64, 100, 0);
    if (error) {
        std::cerr << "Failed to set font size!(" << error << ")\n";;
        exit(-1);
    }
}

void finish(FT_Library &library, FT_Face &face) {
    FT_Done_Face(face);
    FT_Done_FreeType(library);
}

int render_single_character_on_array(unsigned char *array, 
    unsigned int height, unsigned int width, FT_Bitmap *bitmap, 
    unsigned int x, unsigned int y, unsigned int bearingX=0) {
    /** 
     * NOTE: Currently, overwrites the bounding box of the character.
     * To turn this behavior off, the assignment operator for the array
     * should be changed to an OR (|). 
     * Returns 0 on success, -1 on exceeding width, -2 on exceeding height.
     */
    unsigned int bwidth = bitmap->width, brows = bitmap->rows;
    if (bearingX > x)
        bearingX = 0;
    if (x + bwidth > width + bearingX)
        return -1;
    else if (y + brows > height) {
        // return -2; // This cuts off some characters in the last
                      // line -- instead partially render the character
                      // if possible
        if (y >= height) 
            return 0; // Assume succesful - the caller will ensure to break
        brows = height - y;
    }
    
    for (unsigned int i = 0; i < bwidth; i++) {
        for (unsigned int j = 0; j < brows; j++) {
            array[(y+j)*width+(x+i-bearingX)] = 
                bitmap->buffer[j*bwidth+i];
        }
    }
    return 0;
}

std::string render_text(FT_Face &face, std::string text, unsigned char *array, 
    unsigned int height, unsigned int width, unsigned int fontsize, 
    int line_space, bool fixed_width, bool fix_spacing, bool no_partial) {
    int len = text.length();
    int cur = 0;
    FT_UInt glyph_index;
    FT_GlyphSlot slot = face->glyph;

    unsigned int vertical_stride = fontsize;
    if (line_space == -1)
        vertical_stride += (fontsize+1)/2;
    else
        vertical_stride += ((unsigned int)line_space);
    unsigned int maxhoriY = 0; //, minhoriY = 100;
    unsigned int widths[len];

    if (fixed_width) {
        std::unordered_map<char, unsigned int> cache;
        for (int i = 0; i < len; i++) {
            if (cache.find(text[i]) != cache.end()) {
                widths[i] = cache[text[i]];
            } else {
                glyph_index = FT_Get_Char_Index(face, text[i]);
                FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
                widths[i] = slot->advance.x >> 6;
                cache[text[i]] = widths[i];
                unsigned int horiY = ((unsigned int)
                    slot->metrics.horiBearingY) >> 6;
                if (horiY <= 50) 
                    maxhoriY = std::max(maxhoriY, horiY);
                // if (horiY > 0)
                //     minhoriY = std::min(horiY, minhoriY);
            }
        }
    } else {
        for (int i = 0; i < len; i++) {
            glyph_index = FT_Get_Char_Index(face, text[i]);
            FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
            widths[i] = slot->advance.x >> 6;
            unsigned int horiY = ((unsigned int)slot->metrics.horiBearingY) >> 6;
            if (horiY <= 50)
                maxhoriY = std::max(maxhoriY, horiY);
            // if (horiY > 0)
            //     minhoriY = std::min(horiY, minhoriY);
        }
    }

    memset(array, 0, height*width);
    unsigned int leftmargin = 3;
    unsigned int topmargin = 3;
    unsigned int x = leftmargin;
    unsigned int y = topmargin;

    unsigned int lasthoriX = 0;

    while (cur < len) {
        if (text[cur] == '\n') {
            // Just move to the beginning of the new line
            y += vertical_stride;
            x = leftmargin;
            cur++;
            continue;
        }

        // If the previous character was a space and the current one is
        // not, we need to decide now if we should move to the next line.
        if (cur && (text[cur-1] == ' ' || text[cur-1] == '\t')
            && (text[cur] != ' ' && text[cur] != '\t')) {
            unsigned int cur_width = 0;
            for (int i = cur; i < len && text[i] != ' ' && 
                text[i] != '\t'; i++) {
                cur_width += widths[i];
                if (cur_width > width)
                    break;
            }
            if (x+cur_width > width) {
                y += vertical_stride;
                x = leftmargin;
            }
        }

        // Try rendering the current character
        glyph_index = FT_Get_Char_Index(face, text[cur]);
        if (FT_Load_Glyph(face, glyph_index, FT_LOAD_RENDER)) {
            // Can't render - move on
            cur++;
            continue;
        }

        unsigned int horiY = 
            ((unsigned int)slot->metrics.horiBearingY) >> 6;
        unsigned int yoffset = maxhoriY - 
            ((horiY > 50)? maxhoriY : horiY);

        unsigned int compY = (no_partial? y + maxhoriY : y);
        if (compY >= height)
            break;
        
        unsigned int bearingX;
        if (fix_spacing) {
            unsigned int horiX = 
                ((unsigned int)slot->metrics.horiBearingX) >> 6;
            if (horiX > 100)
                horiX = 0;
            lasthoriX = std::max(lasthoriX, horiX);
            bearingX = lasthoriX - horiX;
        } else 
            bearingX = 0;

        int status = render_single_character_on_array(array, height, 
            width, &slot->bitmap, x, y + yoffset, bearingX);
        if (status == -1) {
            // Exceeding the specified width, try again on a new line
            y += vertical_stride;
            x = leftmargin;
            status = render_single_character_on_array(array, height, 
                width, &slot->bitmap, x, y + yoffset, bearingX);
        }

        if (status == 0) {
            // We succeeded, so update (otherwise we skip the character)
            x += slot->advance.x >> 6;
            y += slot->advance.y >> 6; // zero
            if (x > width) {
                x = leftmargin;
                y += vertical_stride;
            }
        }
        cur++;
    }

    return text.substr(0, cur);
}

std::string render_text_unicode(FT_Face &face, std::string text, 
    unsigned char *array, unsigned int height, unsigned int width, 
    unsigned int fontsize, int line_space, bool fixed_width, 
    bool fix_spacing, bool no_partial, bool no_margin, bool fixed_offset) {
    int len = text.length();
    int cur = 0;
    FT_UInt glyph_index;
    FT_GlyphSlot slot = face->glyph;

    unsigned int vertical_stride = fontsize;
    if (line_space == -1)
        vertical_stride += (fontsize+1)/2;
    else
        vertical_stride += ((unsigned int)line_space);
    unsigned int maxhoriY = 0; //, minhoriY = 100;
    unsigned int widths[len];

    unsigned char *src = (unsigned char *)text.c_str();
    unsigned int decoded[len];
    int cur_e = 0, cur_d = 0;
    while (cur_e < len) {
        if (src[cur_e] >> 7 == 0) 
            decoded[cur_d++] = src[cur_e++];
        else if (((src[cur_e] >> 5)&1) == 0) {
            decoded[cur_d++] = 
                ((src[cur_e] & 31) << 6) | (src[cur_e+1] & 63);
            cur_e += 2;
        } else if (((src[cur_e] >> 4)&1) == 0) {
            decoded[cur_d++] = 
                ((src[cur_e] & 15) << 12) | 
                ((src[cur_e+1] & 63) << 6) | 
                (src[cur_e+2] & 63);
            cur_e += 3;
        } else {
            decoded[cur_d++] = 
                ((src[cur_e] & 7) << 18) | 
                ((src[cur_e+1] & 63) << 12) | 
                ((src[cur_e+2] & 63) << 6) |
                (src[cur_e+3] & 63);
            cur_e += 4;
        }
    }

    len = cur_d;
    FT_Select_Charmap(face, FT_ENCODING_UNICODE);

    if (fixed_width) {
        std::unordered_map<unsigned int, unsigned int> cache;
        for (int i = 0; i < len; i++) {
            if (cache.find(decoded[i]) != cache.end()) {
                widths[i] = cache[decoded[i]];
            } else {
                glyph_index = FT_Get_Char_Index(face, decoded[i]);
                FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
                widths[i] = slot->advance.x >> 6;
                cache[decoded[i]] = widths[i];
                unsigned int horiY = ((unsigned int)
                    slot->metrics.horiBearingY) >> 6;
                if (horiY <= 50) 
                    maxhoriY = std::max(maxhoriY, horiY);
                // if (horiY > 0)
                //     minhoriY = std::min(minhoriY, horiY);
            }
        }
    } else {
        for (int i = 0; i < len; i++) {
            glyph_index = FT_Get_Char_Index(face, decoded[i]);
            FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
            widths[i] = slot->advance.x >> 6;
            unsigned int horiY = ((unsigned int)slot->metrics.horiBearingY) >> 6;
            if (horiY <= 50)
                maxhoriY = std::max(maxhoriY, horiY);
            // if (horiY > 0)
            //     minhoriY = std::min(minhoriY, horiY);
        }
    }

    memset(array, 0, height*width);
    unsigned int leftmargin = 3;
    unsigned int topmargin = 3;
    if (no_margin) {
        leftmargin = 0;
        topmargin = 0;
    }
    unsigned int x = leftmargin;
    unsigned int y = topmargin;

    unsigned int lasthoriX = 0;
    cur_e = 0;

    if (fixed_offset) {
        // We fixed the maxhoriY to make sure there is no shift
        // We test the best maxhoriY by using the following sequence:
        // 你好abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,!?./\\-+=@#$%^&*()[]{}<>~`'\";:|_
        // we found that between font size 6-12 (PIXEL uses equivalent to 10) 
        // maxhoriY = fontsize + 2
        if ((fontsize <= 12) and (fontsize >= 6)) {
            maxhoriY = fontsize + 2;
        } else {
            if (fontsize < 6) {
                maxhoriY = fontsize + 1;
            }
            if (fontsize > 12) {
                maxhoriY = fontsize + 3;
            }
        }
    }

    while (cur < len) { 
        if (decoded[cur] == 0xA) {
            // Just move to the beginning of the new line
            y += vertical_stride;
            x = leftmargin;
            cur++;
            cur_e++;
            while ((src[cur_e] & 0xc0) == 0x80)
                cur_e++;
            continue;
        }

        // If the previous character was a space and the current one is
        // not, we need to decide now if we should move to the next line.
        if (cur && (decoded[cur-1] == 0x20 || decoded[cur-1] == 0x9)
            && (decoded[cur] != 0x9 && decoded[cur] != 0x9)) {
            unsigned int cur_width = 0;
            for (int i = cur; i < len && decoded[i] != 0x20 && 
                decoded[i] != 0x9; i++) {
                cur_width += widths[i];
                if (cur_width > width)
                    break;
            }
            if (x+cur_width > width) {
                y += vertical_stride;
                x = leftmargin;
            }
        }

        // Try rendering the current character
        glyph_index = FT_Get_Char_Index(face, decoded[cur]);
        if (FT_Load_Glyph(face, glyph_index, FT_LOAD_RENDER)) {
            // Can't render - move on
            cur++;
            cur_e++;
            while ((src[cur_e] & 0xc0) == 0x80)
                cur_e++;
            continue;
        }

        unsigned int horiY = 
            ((unsigned int)slot->metrics.horiBearingY) >> 6;
        unsigned int yoffset = maxhoriY - 
            ((horiY > maxhoriY)? maxhoriY : horiY);

        unsigned int bearingX;
        if (fix_spacing) {
            unsigned int horiX = 
                ((unsigned int)slot->metrics.horiBearingX) >> 6;
            if (horiX > 100)
                horiX = 0;
            lasthoriX = std::max(lasthoriX, horiX);
            bearingX = lasthoriX - horiX;
        } else 
            bearingX = 0;

        unsigned int compY = (no_partial? y + maxhoriY : y);
        if (compY > height)
            break;

        int status = render_single_character_on_array(array, height, 
            width, &slot->bitmap, x, y + yoffset, bearingX);
        
        if (status == -1) {
            // Exceeding the specified width, try again on a new line
            y += vertical_stride;
            x = leftmargin;
            status = render_single_character_on_array(array, height, 
                width, &slot->bitmap, x, y + yoffset, bearingX);
        }

        if (status == 0) {
            // We succeeded, so update (otherwise we skip the character)
            x += slot->advance.x >> 6;
            y += slot->advance.y >> 6; // zero
            if (x > width) {
                x = leftmargin;
                y += vertical_stride;
            }
        }

        cur++;
        cur_e++;
        while ((src[cur_e] & 0xc0) == 0x80)
            cur_e++;
    }

    return text.substr(0, cur_e);
}

void only_render_text_unicode(FT_Face &face, std::string text, 
    unsigned char *array, unsigned int height, unsigned int width, 
    unsigned int fontsize, int line_space, bool fixed_width, 
    bool fix_spacing, bool no_partial) {
    int len = text.length();
    int cur = 0;
    FT_UInt glyph_index;
    FT_GlyphSlot slot = face->glyph;

    unsigned int vertical_stride = fontsize;
    if (line_space == -1)
        vertical_stride += (fontsize+1)/2;
    else
        vertical_stride += ((unsigned int)line_space);
    unsigned int maxhoriY = 0; //, minhoriY = 100;
    unsigned int widths[len];

    unsigned char *src = (unsigned char *)text.c_str();
    unsigned int decoded[len];
    int cur_e = 0, cur_d = 0;
    while (cur_e < len) {
        if (src[cur_e] >> 7 == 0) 
            decoded[cur_d++] = src[cur_e++];
        else if (((src[cur_e] >> 5)&1) == 0) {
            decoded[cur_d++] = 
                ((src[cur_e] & 31) << 6) | (src[cur_e+1] & 63);
            cur_e += 2;
        } else if (((src[cur_e] >> 4)&1) == 0) {
            decoded[cur_d++] = 
                ((src[cur_e] & 15) << 12) | 
                ((src[cur_e+1] & 63) << 6) | 
                (src[cur_e+2] & 63);
            cur_e += 3;
        } else {
            decoded[cur_d++] = 
                ((src[cur_e] & 7) << 18) | 
                ((src[cur_e+1] & 63) << 12) | 
                ((src[cur_e+2] & 63) << 6) |
                (src[cur_e+3] & 63);
            cur_e += 4;
        }
    }

    len = cur_d;
    FT_Select_Charmap(face, FT_ENCODING_UNICODE);

    if (fixed_width) {
        std::unordered_map<unsigned int, unsigned int> cache;
        for (int i = 0; i < len; i++) {
            if (cache.find(decoded[i]) != cache.end()) {
                widths[i] = cache[decoded[i]];
            } else {
                glyph_index = FT_Get_Char_Index(face, decoded[i]);
                FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
                widths[i] = slot->advance.x >> 6;
                cache[decoded[i]] = widths[i];
                unsigned int horiY = ((unsigned int)
                    slot->metrics.horiBearingY) >> 6;
                if (horiY <= 50) 
                    maxhoriY = std::max(maxhoriY, horiY);
                // if (horiY > 0)
                //     minhoriY = std::min(minhoriY, horiY);
            }
        }
    } else {
        for (int i = 0; i < len; i++) {
            glyph_index = FT_Get_Char_Index(face, decoded[i]);
            FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_BITMAP);
            widths[i] = slot->advance.x >> 6;
            unsigned int horiY = ((unsigned int)slot->metrics.horiBearingY) >> 6;
            if (horiY <= 50)
                maxhoriY = std::max(maxhoriY, horiY);
            // if (horiY > 0)
            //     minhoriY = std::min(minhoriY, horiY);
        }
    }

    memset(array, 0, height*width);
    unsigned int leftmargin = 3;
    unsigned int topmargin = 3;
    unsigned int x = leftmargin;
    unsigned int y = topmargin;

    unsigned int lasthoriX = 0;

    while (cur < len) { 
        if (decoded[cur] == 0xA) {
            // Just move to the beginning of the new line
            y += vertical_stride;
            x = leftmargin;
            cur++;
            continue;
        }

        // If the previous character was a space and the current one is
        // not, we need to decide now if we should move to the next line.
        if (cur && (decoded[cur-1] == 0x20 || decoded[cur-1] == 0x9)
            && (decoded[cur] != 0x9 && decoded[cur] != 0x9)) {
            unsigned int cur_width = 0;
            for (int i = cur; i < len && decoded[i] != 0x20 && 
                decoded[i] != 0x9; i++) {
                cur_width += widths[i];
                if (cur_width > width)
                    break;
            }
            if (x+cur_width > width) {
                y += vertical_stride;
                x = leftmargin;
            }
        }

        // Try rendering the current character
        glyph_index = FT_Get_Char_Index(face, decoded[cur]);
        if (FT_Load_Glyph(face, glyph_index, FT_LOAD_RENDER)) {
            // Can't render - move on
            cur++;
            continue;
        }

        unsigned int horiY = 
            ((unsigned int)slot->metrics.horiBearingY) >> 6;
        unsigned int yoffset = maxhoriY - 
            ((horiY > 50)? maxhoriY : horiY);

        unsigned int bearingX;
        if (fix_spacing) {
            unsigned int horiX = 
                ((unsigned int)slot->metrics.horiBearingX) >> 6;
            if (horiX > 100)
                horiX = 0;
            lasthoriX = std::max(lasthoriX, horiX);
            bearingX = lasthoriX - horiX;
        } else 
            bearingX = 0;

        unsigned int compY = (no_partial? y + maxhoriY : y);
        if (compY > height)
            break;

        int status = render_single_character_on_array(array, height, 
            width, &slot->bitmap, x, y + yoffset, bearingX);
        
        if (status == -1) {
            // Exceeding the specified width, try again on a new line
            y += vertical_stride;
            x = leftmargin;
            status = render_single_character_on_array(array, height, 
                width, &slot->bitmap, x, y + yoffset, bearingX);
        }

        if (status == 0) {
            // We succeeded, so update (otherwise we skip the character)
            x += slot->advance.x >> 6;
            y += slot->advance.y >> 6; // zero
            if (x > width) {
                x = leftmargin;
                y += vertical_stride;
            }
        }

        cur++;
    }
}


void show_image_in_terminal(unsigned char *image, unsigned int height, 
    unsigned int width) {
    unsigned int i, j;
    for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++)
        putchar(image[i*width+j] == 0 ? ' ' : 
            (image[i*width+j] < 128 ? '+' : '*'));
    putchar('\n');
    }
}

std::string render_text_onto_array(std::string text, unsigned char *array, 
    int height, int width, int fontsize, int line_space, bool fixed_width,
    bool fix_spacing, bool no_partial, std::string fontpath) {
    FT_Library library;
    FT_Face face;

    init_ft(library);
    init_face(library, face, fontpath, fontsize);

    std::string rendered = render_text(face, text, (unsigned char *)array, 
        (unsigned int)height, (unsigned int)width, (unsigned int)fontsize, 
        line_space, fixed_width, fix_spacing, no_partial);

    finish(library, face);
    return rendered;
}

std::string render_text_onto_array_unicode(std::string text, unsigned char *array, 
    int height, int width, int fontsize, int line_space, bool fixed_width,
    bool fix_spacing, bool no_partial, bool no_margin, bool fixed_offset, std::string fontpath) {
    FT_Library library;
    FT_Face face;

    init_ft(library);
    init_face(library, face, fontpath, fontsize);

    std::string rendered = render_text_unicode(face, text, 
        (unsigned char *)array, (unsigned int)height, (unsigned int)width,
        (unsigned int)fontsize, line_space, fixed_width, fix_spacing, 
        no_partial, no_margin, fixed_offset);

    finish(library, face);
    return rendered;
}

void only_render_text_onto_array_unicode(std::string text, 
    unsigned char *array, int height, int width, int fontsize, 
    int line_space, bool fixed_width, bool fix_spacing, bool no_partial,
    std::string fontpath) {
    FT_Library library;
    FT_Face face;

    init_ft(library);
    init_face(library, face, fontpath, fontsize);

    only_render_text_unicode(face, text, (unsigned char *)array, 
        (unsigned int)height, (unsigned int)width, (unsigned int)fontsize, 
        line_space, fixed_width, fix_spacing, no_partial);

    finish(library, face);
}