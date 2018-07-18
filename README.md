# leaf-image-segmentation

Based on [Automatic Leaf Extraction from Outdoor
Images ](https://arxiv.org/pdf/1709.06437.pdf)

Steps Finished 
1. Background marker generation




Note : currently we can generate marker for plant leaf image 

Use the following command to segment image
```
usage: generate_marker.py [-h] [-c] [-u] [-s] [-d DESTINATION] image_file

positional arguments:
  image_file            An image filename with its full path

optional arguments:
  -h, --help            show this help message and exit
  -c, --contrast        The image will be output as black background
                        and white foreground
  -u, --unfilled        Output unfilled image with holes
  -s, --smooth          Output image with smooth edges
  -d DESTINATION, --destination DESTINATION
                        Destination directory for the output image
                        If not specified destination directory will
                        be input image directory
```

#### Example:

__Command used__: `python3 generate_marker.py -u -s`

__Input Image__
        
![alt Healthy Apple Leaf](test_images/apple_healthy.JPG)

__Output Image__

![alt Segmented Healthy Apple Leaf](test_images/apple_healthy_marked.JPG)
