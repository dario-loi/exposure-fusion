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
    for element in path_list:
        listbox.insert(0 ,element)
    return

window = Tk()

window.geometry("800x600")
window.configure(bg = "#FFFFFF")

#Main frame
main_frame = Canvas(master = window, bg = "#FFFFFF", height = 600, width = 800, bd = 0, highlightthickness = 0, relief = "ridge")
main_frame.place(x = 0, y = 0)
main_frame.create_rectangle(500.0, 0.0, 800.0, 600.0, fill="#FFFFFF", outline="")

#Title
main_frame.create_text( 570, 30.0, anchor="nw",text="Exposure \n  fusions", fill="#000000", font=("Inter Bold", 38 * -1))

#Listbox path
listbox = Listbox(window)
listbox.place(x=550.0, y=130.0, width=200.0, height=150.0)

#Remove button, on click remove a path from the listbox-
remove_button_image = PhotoImage(file=relative_to_assets("button_3.png"))
remove_button = Button(image=remove_button_image, borderwidth=0, highlightthickness=0, command=lambda: print("delete"), relief="flat")
remove_button.place(x=550.0, y=310.0, width=200.0, height=50.0)


#Upload button, on click open task to select file-
upload_button_image = PhotoImage(file=relative_to_assets("button_1.png"))
upload_button = Button(image=upload_button_image, borderwidth=0, highlightthickness=0, command=select_file, relief="flat")
upload_button.place(x=550.0, y=380.0, width=200.0, height=50.0)


#Line
main_frame.create_line(550.0, 460.0, 750.0, 460.0 , fill="#000000", width=1.0)


#Execute button, on click execute the task-
execute_button_image = PhotoImage(file=relative_to_assets("button_2.png"))
execute_button = Button(image=execute_button_image, borderwidth=0, highlightthickness=0, command=lambda: print("button_2 clicked"),relief="flat")
execute_button.place(x=550.0,y=490.0,width=200.0,height=50.0)


#Text
main_frame.create_text(540.0, 575.0, anchor="nw", text="©Copyright 2022 Gezzi Flavio and Loi Dario", fill="#000000",font=("Inter", 12 * -1))


#Slider Frame
slider_frame = Canvas(window,bg = "#FFFFFF",height = 600,width = 500,bd = 0,highlightthickness = 0,relief = "ridge")
slider_frame.place(x = 0, y = 0)
app = Application(master=slider_frame)


def build():  
    window.resizable(False, False)
    window.title("Exposure fusions")
    window.mainloop()

build()
