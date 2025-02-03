from tkinter import *
import tkinter as tk

def screen4(fourthpage):
    fourthpage.title("VIT STUDENT BANK")
    fourthpage.geometry("700x500+200+150")
    tqlabel = Label(fourthpage, text="Thank you... for banking with us", font=("Times_New_Roman", 25))
    tqlabel.place(x=120, y=70)

def page4():
    fourthpage = Tk()
    screen4(fourthpage)
    fourthpage.mainloop()
