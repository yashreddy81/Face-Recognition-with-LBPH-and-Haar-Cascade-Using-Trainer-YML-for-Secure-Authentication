from tkinter import *
from page2 import *

def change(mainpage):
    mainpage.destroy()
    page2()

def start(mainpage):
    mainpage.title("VIT STUDENT BANK")
    mainpage.geometry("700x500+200+150")
    l1 = Label(mainpage, text='Welcome to VIT STUDENT BANK', font=("Times_New_Roman", 25))
    l1.place(x=120, y=70)
    button1 = Button(mainpage, text="Next page", width=20, command=lambda: change(mainpage))
    button1.place(x=500, y=450)

def main():
    mainpage = Tk()
    start(mainpage)
    mainpage.mainloop()

main()
