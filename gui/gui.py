# Description: This file contains the code for the GUI of the application
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import Listbox
from image_slider import *


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"elements")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def select_file():
    path_list = []
    filetypes = (('images files', '*.jpg'),('All files', '*.*'))

    path_list += fd.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    return path_list

window = Tk()

window.geometry("800x600")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 600,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    500.0,
    0.0,
    800.0,
    600.0,
    fill="#FFFFFF",
    outline="")

canvas.create_text(
    570,
    30.0,
    anchor="nw",
    text="Exposure \n  fusions",
    fill="#000000",
    font=("Inter Bold", 38 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=select_file,
    relief="flat"
)
button_1.place(
    x=550.0,
    y=175.0,
    width=200.0,
    height=50.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=550.0,
    y=440.0,
    width=200.0,
    height=50.0
)

canvas.create_text(
    540.0,
    575.0,
    anchor="nw",
    text="©Copyright 2022 Gezzi Flavio and Loi Dario",
    fill="#000000",
    font=("Inter", 12 * -1)
)

canvas.create_rectangle(
    549.0,
    399.0,
    750.0,
    400.0,
    fill="#000000",
    outline="")

###############
#    Create a rectangle
#    x = 0, y = 0, width = 500, height = 600
###############
canvas.create_rectangle(

    0.0,
    0.0,
    500.0,
    600.0,
    fill="#D9D9D9",
    outline="")

rettangoloslider = Canvas(
    window,
    bg = "#FFFFFF",
    height = 600,
    width = 500,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
rettangoloslider.place(x = 0, y = 0)
app = Application(master=rettangoloslider)


#listbox = Listbox(window, width=20, height=10)  

#listbox.insert(1,"path1")  
#listbox.insert(2, "path2")   

#listbox.pack()

def build():  
    window.resizable(False, False)
    window.title("Exposure fusions")
    window.mainloop()

build()
