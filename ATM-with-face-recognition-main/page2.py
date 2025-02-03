from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
from opencvwebcam import detect_face 
from page3 import *

def login(mainpage):
    username = username_entry.get()
    password = password_entry.get()

    if username and password:
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="password", database="bankdb")
        cursor = mydb.cursor()
        selectquery = "SELECT * FROM users WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery, (username, password))
        results = cursor.fetchall()

        if results:
            if detect_face(username): 
                messagebox.showinfo("Success", "Face recognized. Access granted.")
                mainpage.destroy()
                page3(username, password) 
            else:
                messagebox.showerror("Error", "Face not recognized. Access denied.")
        else:
            messagebox.showerror("Error", "Invalid username or password.")

        cursor.close()
        mydb.close()
    else:
        messagebox.showwarning("Warning", "Please enter both username and password.")

def screen2(mainpage):
    mainpage.title("VIT STUDENT BANK")
    mainpage.geometry("700x500+200+150")

    label1 = Label(mainpage, text='Login to your account', font=("Times_New_Roman", 25))
    label1.place(x=180, y=70)

    global username_entry
    global password_entry

    username_label = Label(mainpage, text='Username:', font=("Times_New_Roman", 12))
    username_label.place(x=200, y=150)
    username_entry = Entry(mainpage, font=("Times_New_Roman", 12))
    username_entry.place(x=300, y=150)

    password_label = Label(mainpage, text='Password:', font=("Times_New_Roman", 12))
    password_label.place(x=200, y=200)
    password_entry = Entry(mainpage, font=("Times_New_Roman", 12), show='*')
    password_entry.place(x=300, y=200)

    login_button = Button(mainpage, text='Login', command=lambda: login(mainpage), width=20)
    login_button.place(x=300, y=250)

def page2():
    mainpage = Tk()
    screen2(mainpage)
    mainpage.mainloop()
