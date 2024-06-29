# Mini-Photoshop
 

## General Overview

This Mini Photoshop application is a simplified photo editing tool designed for basic image manipulation tasks, similar to those found in professional software like Adobe Photoshop. The application features a user-friendly interface allowing users to apply various effects to images, including grayscale conversion, image sharpening, brightness adjustment, and custom BMP parsing for image loading. It supports a range of image formats such as BMP, JPEG, and PNG.

## Language and Tools

- **Language:** Python
- **GUI:** Tkinter
- **Image Handling:** PIL (Python Imaging Library)
- **Numerical Operations:** NumPy

## Functionalities

### Core Operations

- **Open File:** Load images from the file system. Supports .jpeg/.jpg/.png/.bmp formats.
- **Exit:** Closes the GUI and terminates the program.

### Image Effects

- **Greyscale:** Converts color images to grayscale, simulating a black-and-white photograph.
- **Ordered Dithering:** Introduces a dot pattern into the image, simulating different tones and shades with a limited color palette.
- **Huffman Coding:** Provides lossless data compression, showing statistics like entropy and average code length.
- **Auto Level:** Automatically adjusts the contrast of an image, enhancing its overall appearance.

### Other Operations

- **Mirror:** Provides the mirror image of the input image.
- **Sketch:** Converts the input image into a pencil sketch.
- **Sharpening:** Removes blur from the image, enhancing its quality.
- **Brighten:** Increases the brightness of the image by 80%.
- **Passport Size:** Converts the image to a passport size (120x120 px).

## Running the Application

### Using the Executable

1. Locate the `dist` folder.
2. Run `app.exe` (Windows only).

### Installing the Requirements and Running the File

1. Install required libraries:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the application:
    ```sh
    python app.py
    ```
    
## Conclusion

This Mini Photoshop application, with its blend of practical image editing capabilities and insights into image processing algorithms, provides a unique tool for users and learners alike. It exemplifies the power of Python and its libraries in creating functional and educational software applications.
