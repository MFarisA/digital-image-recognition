from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Functionality placeholders
def open_image():
    global img, img_display, img_path, original_size
    file_path = filedialog.askopenfilename(
        initialdir="data/",
        title="Select an Image File",
        filetypes=[("All Files", "*.*")]
    )

    if file_path:
        try:
            img_path = file_path
            img = Image.open(file_path).convert("RGBA")
            img_display = ImageTk.PhotoImage(img.resize((250, 250)))
            label_original_img.config(image=img_display)
            label_original_img.image = img_display  # Prevent garbage collection
            original_size = os.path.getsize(file_path) / 1024  # Size in KB
            entry_size_original.delete(0, END)
            entry_size_original.insert(0, f"{original_size:.4f} kb")
        except Exception as e:
            print("Error opening image:", e)

def compress_image():
    global compressed_img, compressed_size, img_display_compressed
    try:
        img_cv = cv2.imread(img_path)

        # Ensure the directory exists
        output_dir = "data/result-compressed"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the path for the compressed image
        compressed_path = os.path.join(output_dir, "compressed_image.jpg")

        # Compress the image and save it
        cv2.imwrite(compressed_path, img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # Compression quality 50%
        
        compressed_size = os.path.getsize(compressed_path) / 1024  # Size in KB
        compressed_img = Image.open(compressed_path).resize((250, 250))
        img_display_compressed = ImageTk.PhotoImage(compressed_img)
        label_compressed_img.config(image=img_display_compressed)
        label_compressed_img.image = img_display_compressed

        entry_size_compressed.delete(0, END)
        entry_size_compressed.insert(0, f"{compressed_size:.4f} kb")

        calculate_metrics(img_path, compressed_path)
        plot_histogram()
    except Exception as e:
        print("Error compressing image:", e)

def calculate_metrics(original, compressed):
    global mse_value, psnr_value, ssim_value, entropy_value
    
    original_img = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
    compressed_img = cv2.imread(compressed, cv2.IMREAD_GRAYSCALE)

    mse_value = np.mean((original_img - compressed_img) ** 2)
    psnr_value = cv2.PSNR(original_img, compressed_img)
    ssim_value, _ = ssim(original_img, compressed_img, full=True)
    entropy_value = -np.sum(original_img / 255 * np.log2(original_img / 255 + 1e-10))

    entry_psnr.delete(0, END)
    entry_psnr.insert(0, f"{psnr_value:.4f}")

    entry_mse.delete(0, END)
    entry_mse.insert(0, f"{mse_value:.4f}")

    entry_ssim.delete(0, END)
    entry_ssim.insert(0, f"{ssim_value:.4f}")

    entry_entropy.delete(0, END)
    entry_entropy.insert(0, f"{entropy_value:.4f}")

def reset_fields():
    label_original_img.config(image="", bg="gray")
    entry_size_original.delete(0, END)
    label_compressed_img.config(image="", bg="white")
    entry_size_compressed.delete(0, END)
    entry_psnr.delete(0, END)
    entry_mse.delete(0, END)
    entry_ssim.delete(0, END)
    entry_entropy.delete(0, END)
    for widget in histogram_frame.winfo_children():
        widget.destroy()

def plot_histogram():
    for widget in histogram_frame.winfo_children():
        widget.destroy()

    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(img_cv.ravel(), 256, [0, 256])
    ax.set_title('Histogram')
    ax.set_xlabel('Pixel Values')
    ax.set_ylabel('Frequency')

    canvas = FigureCanvasTkAgg(fig, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Root window setup
root = Tk()
root.title("Image Compression Tool")
root.geometry("1200x800")
root.resizable(False, False)

# Frames for layout
frame_left = Frame(root, padx=10, pady=10, bg="#f0f0f0", relief=GROOVE, bd=2)
frame_left.pack(side=LEFT, fill=Y)

frame_center = Frame(root, padx=10, pady=10, bg="#f9f9f9", relief=GROOVE, bd=2)
frame_center.pack(side=LEFT, fill=BOTH, expand=True)

frame_right = Frame(root, padx=10, pady=10, bg="#f0f0f0", relief=GROOVE, bd=2)
frame_right.pack(side=RIGHT, fill=Y)

histogram_frame = Frame(root, padx=10, pady=10, bg="#ffffff", relief=GROOVE, bd=2)
histogram_frame.pack(side=BOTTOM, fill=BOTH, expand=True)

# Buttons - Left Frame
Label(frame_left, text="Pengolahan", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=10)
Button(frame_left, text="Buka Citra", width=15, command=open_image).pack(pady=5)
Button(frame_left, text="Kompresi", width=15, command=compress_image).pack(pady=5)
Button(frame_left, text="Reset", width=15, command=reset_fields).pack(pady=5)

# Original Image - Center Frame
Label(frame_center, text="Citra Asli", font=("Arial", 10, "bold"), bg="#f9f9f9").pack(pady=10)
label_original_img = Label(frame_center, width=250, height=250, bg="gray")
label_original_img.pack(pady=5)
Label(frame_center, text="Ukuran Citra Asli", bg="#f9f9f9").pack()
entry_size_original = Entry(frame_center, width=30)
entry_size_original.pack(pady=5)

# Compressed Image - Right Frame
Label(frame_right, text="Citra Hasil Kompresi", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=10)
label_compressed_img = Label(frame_right, width=250, height=250, bg="white")
label_compressed_img.pack(pady=5)
Label(frame_right, text="Ukuran Citra Hasil Kompresi", bg="#f0f0f0").pack()
entry_size_compressed = Entry(frame_right, width=30)
entry_size_compressed.pack(pady=5)

Label(frame_right, text="PSNR", bg="#f0f0f0").pack()
entry_psnr = Entry(frame_right, width=30)
entry_psnr.pack(pady=5)

Label(frame_right, text="MSE", bg="#f0f0f0").pack()
entry_mse = Entry(frame_right, width=30)
entry_mse.pack(pady=5)

Label(frame_right, text="SSIM", bg="#f0f0f0").pack()
entry_ssim = Entry(frame_right, width=30)
entry_ssim.pack(pady=5)

Label(frame_right, text="Entropy", bg="#f0f0f0").pack()
entry_entropy = Entry(frame_right, width=30)
entry_entropy.pack(pady=5)

# Run the application
root.mainloop()
