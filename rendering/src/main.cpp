#include <iostream>
#include "render.h"

#define HEIGHT 256
#define WIDTH 512

using namespace std;

int main() {
    FT_Library library;
    FT_Face face;

    init_ft(library);
    init_face(library, face);

    string teststring =  "This is a placeholder for the main code.";
    unsigned char array[HEIGHT*WIDTH];

    render_text(face, teststring, (unsigned char *)array, HEIGHT, WIDTH);
    show_image_in_terminal(array, HEIGHT, WIDTH);    

    finish(library, face);

    return 0;
}