# Description: This file contains the code for the GUI of the application
from exposure_fusion import ExposureFusion
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import Listbox
from image_slider import *
from os import getcwd
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"elements")

fuser = ExposureFusion(perform_alignment=True, pyramid_levels=3, sigma=0.2)


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def upload_file():
    path_list = []

    supported_ext = ['raw', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']

    ext_regex = [f"*.{ext}" for ext in supported_ext]
    ext_strings = [f"{ext.upper()} files" for ext in supported_ext]
    ext_strings.append("All files")
    ext_regex.append("*.*")
    filetypes = tuple(zip(ext_strings, ext_regex))

    path_list += fd.askopenfilenames(
        title='Open a file',
        initialdir=getcwd(),
        filetypes=filetypes)
    for element in path_list:
        listbox.insert(0, element)
    return


def delete_file():
    listbox.delete(tk.ANCHOR)
    return


def execute_file():

    images = [cv2.imread(elements) for elements in listbox.get(0, tk.END)]

    fuser.align_images = bool(var1.get())

    HDR = fuser(images)

    if HDR is not None:
        cv2.imwrite("data/pictures/HDR_test_scene_1.png", HDR)
    return


window = Tk()
screen_width = window.winfo_screenwidth()  # Width of the screen
screen_height = window.winfo_screenheight()  # Height of the screen

# Calculate Starting X and Y coordinates for Window
x = (screen_width/2) - (800/2)
y = (screen_height/2) - (600/2)

# Set the window size and center in the screen
window.geometry('%dx%d+%d+%d' % (800, 600, x, y))
window.configure(bg="#FFFFFF")


# Main frame
main_frame = Canvas(master=window, bg="#FFFFFF", height=600,
                    width=800, bd=0, highlightthickness=0, relief="ridge")
main_frame.place(x=0, y=0)
main_frame.create_rectangle(
    500.0, 0.0, 800.0, 600.0, fill="#FFFFFF", outline="")

# Title
main_frame.create_text(570, 30.0, anchor="nw", text="Exposure \n  fusions",
                       fill="#000000", font=("Inter Bold", 38 * -1))

# Listbox path
listbox = Listbox(window, background="#FFFFFF", foreground="#000000")
listbox.place(x=550.0, y=130.0, width=200.0, height=150.0)

# Remove button, on click remove a path from the listbox-
remove_button_image = PhotoImage(file=relative_to_assets("button_3.png"))
remove_button = Button(image=remove_button_image, borderwidth=0,
                       highlightthickness=0, command=delete_file, relief="flat")
remove_button.place(x=550.0, y=330.0, width=200.0, height=50.0)


# Upload button, on click open task to select file-
upload_button_image = PhotoImage(file=relative_to_assets("button_1.png"))
upload_button = Button(image=upload_button_image, borderwidth=0,
                       highlightthickness=0, command=upload_file, relief="flat")
upload_button.place(x=550.0, y=390.0, width=200.0, height=50.0)


# Line
main_frame.create_line(550.0, 460.0, 750.0, 460.0, fill="#000000", width=1.0)


# Checkboxes
var1 = tk.IntVar()
c1 = tk.Checkbutton(window, text='Align images', variable=var1,
                    onvalue=1, offvalue=0, background="#FFFFFF", foreground="#000000")
c1.place(x=550.0, y=290.0)

# Execute button, on click execute the task-
execute_button_image = PhotoImage(file=relative_to_assets("button_2.png"))
execute_button = Button(image=execute_button_image, borderwidth=0,
                        highlightthickness=0, command=lambda: execute_file(), relief="flat")
execute_button.place(x=550.0, y=490.0, width=200.0, height=50.0)


# Text
main_frame.create_text(540.0, 575.0, anchor="nw",
                       text="©Copyright 2022 Gezzi Flavio and Loi Dario", fill="#000000", font=("Inter", 12 * -1))

# Slider Frame
slider_frame = Canvas(window, bg="#FFFFFF", height=600,
                      width=500, bd=0, highlightthickness=0, relief="ridge")
slider_frame.place(x=0, y=0)
app = Application(master=slider_frame)


def build():
    window.resizable(False, False)
    window.title("Exposure Fusion")
    window.mainloop()


build()
