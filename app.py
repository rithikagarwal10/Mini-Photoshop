from tkinter import filedialog, messagebox
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
from math import log2
import heapq
import numpy as np
import struct


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.panelA = None
        self.panelB = None
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Mini Photoshop")
        self.root.geometry("1400x800") 
        
        # Title Label
        Label(self.root, text="Rithik's Miniphotoshop", font=('Verdana', 20), pady=10) \
            .pack(side=TOP, fill=X)

        # Frame for images
        self.image_frame = Frame(self.root)
        self.image_frame.pack(expand=True, fill=BOTH, side=TOP)

        # Panel A and B for original and processed images
        self.panelA = Label(self.image_frame, text="Original Image", bg='gray')
        self.panelA.pack(side=LEFT, expand=True, fill=BOTH, padx=10, pady=10)

        self.panelB = Label(self.image_frame, text="Processed Image", bg='gray')
        self.panelB.pack(side=LEFT, expand=True, fill=BOTH, padx=10, pady=10)
        
        # Frame for controls
        self.controls_frame = Frame(self.root, height=50)
        self.controls_frame.pack(side=BOTTOM, fill=X, padx=10, pady=10)

        # Dropdown for core effects
        self.effect_var = StringVar()
        self.effect_var.set("Core Operation")
        effects = ['Grayscale', 'Dither', 'Huffman', 'Auto Level']
        self.effects_combo = ttk.Combobox(self.controls_frame, textvariable=self.effect_var, values=effects, state='readonly', width=25)
        self.effects_combo.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        self.effects_combo.bind('<<ComboboxSelected>>', self.apply_effect)
        
        # Drop down for additional effects
        self.additional_effect_var = StringVar()
        self.additional_effect_var.set("Other Operation")
        additional_effects = ['Mirror','Sketch','Sharpning','Brightness','Passport Size']  # Add other effects as needed
        self.additional_effects_combo = ttk.Combobox(self.controls_frame, textvariable=self.additional_effect_var, values=additional_effects, state='readonly', width=25)
        self.additional_effects_combo.grid(row=0, column=2, padx=10, pady=10, sticky='ew')
        self.additional_effects_combo.bind('<<ComboboxSelected>>', self.apply_additional_effect)

        # Upload Button
        Button(self.controls_frame, text="Upload", command=self.upload, width=25).grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        # Exit Button
        Button(self.controls_frame, text="Exit", command=self.exit_application,width=25).grid(row=0, column=3, padx=10, pady=10, sticky='ew')

        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)


    def apply_effect(self, event=None):
        effect = self.effect_var.get()
        if effect == "Grayscale":
            self.grayscale()
        elif effect == "Dither":
            self.ordered_dithering()
        elif effect == "Huffman":
            self.apply_huffman()
        elif effect == "Auto Level":
            self.apply_auto_level()


    def apply_additional_effect(self, event=None):
        effect = self.additional_effect_var.get()
        if effect == "Mirror":
            self.mirror_image()
        elif effect == "Sketch":
            self.apply_sketch_effect()
        elif effect == "Sharpning":
            self.apply_sharpening()
        elif effect == "Brightness":
            self.increase_brightness()
        elif effect == "Passport Size":
            self.make_passport_size()

    def parse_bmp(self, path):
        with open(path, 'rb') as f:

            header = f.read(14)

            if header[:2] != b'BM':
                raise ValueError('File is not a BMP')

            dib_header = f.read(40)
            width = int.from_bytes(dib_header[4:8], byteorder='little')
            height = int.from_bytes(dib_header[8:12], byteorder='little')
            bit_depth = int.from_bytes(dib_header[14:16], byteorder='little')

            if bit_depth != 24:
                raise ValueError('Only 24-bit BMP files are supported')

            row_size = (bit_depth * width + 31) // 32 * 4
            padding = row_size - (width * 3)

            # Read the pixel data
            pixel_data = []
            for y in range(height):
                row = []
                for x in range(width):
                    b, g, r = f.read(3)
                    row.append((r, g, b))
                pixel_data.append(row)
                f.read(padding)  

            return pixel_data, width, height

    def upload(self):
        f_types = [('All Files', '*.*'), ('Jpg Files', '*.jpg'), ('PNG Files', '*.png'), ('Bmp Files', '*.bmp')]
        path = filedialog.askopenfilename(filetypes=f_types)
        if not path:
            return
        
        # Check the file extension
        file_extension = path.split('.')[-1].lower()
        
        if file_extension == 'bmp':

            try:
                pixel_data, width, height = self.parse_bmp(path)
                # Convert the pixel data to a format suitable for PIL
                self.image = Image.new('RGB', (width, height))
                for y in range(height):
                    for x in range(width):
                        self.image.putpixel((x, y), pixel_data[y][x])
                self.image = self.image.resize((450, 450))  # Resize the image as needed
                self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
                self.image = np.array(self.image)
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return
        else:
            # Use OpenCV for other file types
            self.image = cv2.imread(path)
            image_resized = cv2.resize(self.image, (450, 450))  
            self.image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(self.image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if self.panelA is None:
            self.panelA = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelA.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelA.configure(image=imgtk)
        self.panelA.image = imgtk

    def grayscale(self):

        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return
        
        b, g, r = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        
        # Calculate the grayscale value using the luminance formula
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray.astype(np.uint8)

        img = Image.fromarray(gray)
        imgtk = ImageTk.PhotoImage(image=img)

        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk


    def exit_application(self):
        self.root.destroy()

    def ordered_dithering(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Dither matrix and processing
        dither_matrix = np.array([
            [3, 4, 0],
            [3, 2, 9],
            [1, 8, 5]
        ], dtype=np.uint8)

        for y in range(0, gray.shape[0], 3):
            for x in range(0, gray.shape[1], 3):
                for dy in range(min(3, gray.shape[0] - y)):
                    for dx in range(min(3, gray.shape[1] - x)):
                        pixel_value = gray[y + dy, x + dx]
                        threshold = (dither_matrix[dy, dx] * 255) / 9
                        gray[y + dy, x + dx] = 0 if pixel_value < threshold else 255

        img = Image.fromarray(gray)
        imgtk = ImageTk.PhotoImage(image=img)

        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk


    def apply_huffman(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_data = gray.flatten()

        frequencies = Counter(image_data)
        total_pixels = sum(frequencies.values())

        entropy = -sum((freq / total_pixels) * log2(freq / total_pixels) for freq in frequencies.values() if freq > 0)

        heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_tree = heap[0][1:]

        huffman_codes = {}
        def assign_codes(nodes, prefix=""):
            for node in nodes:
                symbol, code = node
                if isinstance(symbol, list):
                    assign_codes(symbol, prefix + code)
                else:
                    huffman_codes[symbol] = prefix + code

        assign_codes(huffman_tree)

        average_code_length = sum(len(code) * (freq / total_pixels) for symbol, code in huffman_codes.items() for symbol, freq in frequencies.items() if symbol == symbol)

        message = f"Huffman Code Statistics:\nEntropy: {entropy:.2f} bits\nAverage Code Length: {average_code_length:.2f} bits"

        messagebox.showinfo("Huffman Code Statistics", message)



    def apply_auto_level(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        channels = cv2.split(rgb_img)
        new_channels = []

        for channel in channels:
            min_val, max_val = np.percentile(channel, (2, 98))
            LUT = np.interp(np.arange(0, 256), [min_val, max_val], [0, 255]).astype('uint8')
            new_channels.append(cv2.LUT(channel, LUT))

        auto_leveled_img = cv2.merge(new_channels)

        auto_leveled_img_rgb = cv2.cvtColor(auto_leveled_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(auto_leveled_img_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk

    def mirror_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        mirrored_img = cv2.flip(self.image, 1)

        img = Image.fromarray(mirrored_img)
        imgtk = ImageTk.PhotoImage(image=img)

        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk

    def apply_sketch_effect(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return
        
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        inverted_img = cv2.bitwise_not(gray_img)

        blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)

        inverted_blurred_img = cv2.bitwise_not(blurred_img)

        sketch_img = cv2.divide(gray_img, inverted_blurred_img, scale=256.0)
        
        img = Image.fromarray(sketch_img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk

    def apply_sharpening(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])

        sharpened_img = cv2.filter2D(self.image, -1, sharpening_kernel)

        img = Image.fromarray(sharpened_img)
        imgtk = ImageTk.PhotoImage(image=img)

        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk

    def increase_brightness(self, value=90):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        value = np.ones_like(self.image) * value
        value = value.astype(self.image.dtype)
        
        brighter_img = cv2.add(self.image, value)

        img = Image.fromarray(brighter_img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk

    def make_passport_size(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Upload an image before applying effects.")
            return

        height, width = self.image.shape[:2]
        aspect_ratio = width / height
        
        target_height, target_width = 120, 120
        target_aspect_ratio = target_width / target_height
        

        if aspect_ratio > target_aspect_ratio:
            new_width = int(target_aspect_ratio * height)
            x_offset = (width - new_width) // 2
            cropped_img = self.image[0:height, x_offset:x_offset+new_width]
        elif aspect_ratio < target_aspect_ratio:
            new_height = int(width / target_aspect_ratio)
            y_offset = (height - new_height) // 2
            cropped_img = self.image[y_offset:y_offset+new_height, 0:width]
        else:
            cropped_img = self.image

        passport_size_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(passport_size_img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if self.panelB is None:
            self.panelB = Label(self.root, image=imgtk, borderwidth=5, relief="sunken")
            self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=20, pady=20)
        else:
            self.panelB.configure(image=imgtk)
        self.panelB.image = imgtk



if __name__ == "__main__":
    root = Tk()
    app = ImageProcessor(root)
    root.mainloop()
