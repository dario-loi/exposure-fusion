import tkinter as tk
from PIL import Image, ImageTk


class Application(tk.Frame):
    
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_slider()



    def create_slider(self):
        screenHeight = 600
        screemWidth = 500
        # CREATING IMAGES
        
        size = (500, 600)
        self.img1 = ImageTk.PhotoImage(Image.open("gui/elements/test2.png").resize(size))
        self.canv1 = tk.Canvas(self, width=500, height=600, 
            highlightthickness=0, bd=0)
        self.canv1.grid(row=0, column=0, sticky="nsew")
        self.canv1.create_image(500/2, 600/2, image=self.img1, anchor="center")

        self.img2 = ImageTk.PhotoImage(Image.open("gui/elements/test1.png").resize(size))
        self.canv2 = tk.Canvas(self, width=500, height=600, 
            highlightthickness=0, bd=0)
        self.canv2.grid(row=0, column=0, sticky="nsw")
        self.canv2.create_image(500/2, 600/2, image=self.img2, anchor="center")

        self.line = self.canv2.create_line(500/2, 0, 500/2, 600, 
            width=4, fill="white")    

        self.canv2.bind('<B1-Motion>', self.slide_image)
    

    def slide_image(self, event):
        cur_length = int(event.x)
        cur_height = int(event.y)
        if cur_height > ((600/2) - 100) and cur_height < ((600/2) + 100):
            if cur_length < 500: 
                self.canv2.config(width=cur_length)
                self.canv2.coords(self.line, cur_length, 0, cur_length, 600)
            else:
                self.canv2.config(width=500)
                self.canv2.coords(self.line, 500, 0, 500, 600)


def build_slider():
    #Main function to build the slider
    root = tk.Tk()
    root.state('zoomed')
    root.geometry("%dx%d+0+0" % (500, 600))
    root.title("Image Slider")
    app = Application(master=root)
    app.mainloop()
    

