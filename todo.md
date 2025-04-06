# TODO:
    --0 Helper functions
        * Generate array geometries from a grid

        * Make and save to png file
            > void save_to_png(const std::array<float><float> img)

    - 1
        * Output antenna pattern csv for a specific theta (or a specific phi)
            * Python script to take pattern and plot

A) Forward calculation
    1) Take a geometry + complex gains input of some determined format
    and output the pattern as a CSV.
    input file format:
        * N newline separated elements
        * 5 comma separated values per line:
              > (x, y, z) coordinates of array element
              > (mag, phase (rad)) of complex gain

B) Inverse design
    1) Read CSV file with array factor in predescribed format
        Two lines:
        * "t90" (meaning this is the pattern at theta=90
        * comma separated theta/phi values from 0 to 2pi

    Random geometry maker for N elements
    Loss function calculator for comparing array factor with desired AF

## Stretch goals:
    - Also include an element factor for the radiation pattern of each element
    stead of assuming isotropic radiation (perhaps a csv of the element pattern)
    - Can I modify the element pattern per element or section of the array?
    - Live raylib/something output with sliders for
        Number of elements,
        element spacing, and
        frequency
