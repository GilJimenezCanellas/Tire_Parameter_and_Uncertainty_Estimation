''' Helper functions for handling frames in the GUI '''
import tkinter as tk
from PIL import Image, ImageTk

class ConsoleRedirect:
    ''' Redirects stdout to a tkinter ScrolledText widget '''

    def __init__(self, text_widget):
        ''' Initialize the ConsoleRedirect object '''
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        ''' Write the string to the ScrolledText widget '''
        self.buffer += string
        if '\r' in self.buffer:
            self.text_widget.config(state='normal')
            self.text_widget.delete("end-1c linestart", "end-1c lineend")  # Delete the last line
            self.text_widget.insert(tk.END, self.buffer.split('\r')[-1])
            self.text_widget.see(tk.END)
            self.text_widget.update()
            self.text_widget.config(state='disabled')
            self.buffer = ""
        else:
            self.text_widget.config(state='normal')
            self.text_widget.insert(tk.END, string)
            self.text_widget.see(tk.END)
            self.text_widget.update()
            self.text_widget.config(state='disabled')

    def flush(self):
        ''' Flush the output '''
        pass


def load_and_scale_image(image_path, frame):
    ''' Load and scale an image to fit the frame '''
    image = Image.open(image_path)
    frame.update_idletasks()  # Ensure the frame dimensions are updated
    frame_width = frame.winfo_width()
    frame_height = frame.winfo_height()
    image = image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def create_image_canvas(frame, image_path):
    ''' Create a canvas with an image '''
    frame.update_idletasks()  # Ensure the frame dimensions are updated
    icon = load_and_scale_image(image_path, frame)
    image_canvas = tk.Canvas(frame, width=frame.winfo_width(), height=frame.winfo_height())
    image_canvas.create_image(frame.winfo_width() // 2,
                              frame.winfo_height() // 2, anchor=tk.CENTER, image=icon)
    image_canvas.image = icon
    image_canvas.grid(row=0, column=1, sticky="nsew")


def update_grid(event, frames):
    ''' Update the grid layout of the frames '''
    # Dynamic size for every Frame
    for frame in frames:
        total_columns = frame.grid_size()[0]
        for column in range(total_columns):
            frame.grid_columnconfigure(column, weight=1)


def configure_root_grid(root):
    ''' Configure the grid layout of the root window '''
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=4)
    root.grid_rowconfigure(10, weight=1)
