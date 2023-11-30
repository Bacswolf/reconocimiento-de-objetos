from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import imutils
import cv2
from red_predic import Red_pred

imagen = None
filep = None
si = False


def openfile():
    global si
    si = True
    filepath = filedialog.askopenfilename(
        initialdir="/",
        title="Selecciona la imagen.",
        filetypes=(("image files", "*.jpg"), ("all files", "*.*")),
    )

    global imagen
    global filep
    filep = filepath
    imagen = cv2.imread(filepath)
    image = imutils.resize(imagen, height=360)

    imagetoshow = imutils.resize(image, height=120)
    imagetoshow = cv2.cvtColor(imagetoshow, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(imagetoshow)
    imgmostrar = ImageTk.PhotoImage(image=im)
    labelimagen.configure(image=imgmostrar)
    labelimagen.Image = imgmostrar


def clasif():
    if si:
        rd = Red_pred(filep)
        predic2 = rd.predecir()
        textoclase2.config(text=predic2)

    else:
        textoclase.config(text="No has seleccionado ninguna imagen.")


raiz = Tk()

raiz.title("Reconociendo frutas y verduras")
raiz.geometry("700x400")

mframe = Frame()
mframe.pack()
mframe.config(width="700", height="400", bg="black")

labelimagen = Label(mframe)
labelimagen.config(bg="black")
labelimagen.place(x=300, y=50)

label = Button(
    mframe,
    text="Selecciona la imagen.",
    command=openfile,
    bg="white",
    font=("verdana", 12),
)
label.place(x=50, y=50)

botonclasif = Button(
    mframe, text="Clacificar", command=clasif, bg="white", font=("verdana", 12)
)
botonclasif.place(x=50, y=200)

labelclase = Label(
    mframe, text="Clase encontrada:", fg="white", bg="black", font=("verdana", 12)
)
labelclase.place(x=50, y=250)


textoclase2 = Label(mframe)
textoclase2.config(fg="white", bg="black", font=("verdana", 12))
textoclase2.place(x=200, y=270)


raiz.mainloop()
