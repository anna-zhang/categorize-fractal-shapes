# Categorize Fractal Shapes
A tool to categorize images containing fractal shapes. A tool to generate these shapes can be found in [fractal-shapes](https://github.com/anna-zhang/fractal-shapes).

The program takes in standardized images that have the fractal shape centered within the frame as input; these standardized fractal images can be created using the -center flag when generating fractal shapes in [fractal-shapes](https://github.com/anna-zhang/fractal-shapes). All images are resized to a lower dimension to limit potential memory and computation time issues. Images with extremely few white pixels are filtered out into a “shapeless” category. On the remaining images, histogram comparison is used to create initial, high-level shape categories so that similar-weight distributions are grouped together. Within these initial shape categories, images are further grouped using Earth Mover’s Distance as the similarity metric to build final shape categories. 

# Usage - CLI
Give a folder name containing all fractal images to categorize and a resolution to resize each image to before categorizing: <br/>
```python3 combination.py -all (image folder name) (x-resolution) (y-resolution)```<br/>