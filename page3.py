from tkinter import *
import tkinter as tk
from tkinter import messagebox
import mysql.connector
from dbms import *
from page4 import *

username = ""
password = ""

def change(thirdpage):
    thirdpage.destroy()
    page4()

def screen3(thirdpage, username, password):
    global withdrawsuccesslabel
    global depositsuccesslabel
    global balancesuccesslabel
    
    withdrawsuccesslabel = None
    depositsuccesslabel = None
    balancesuccesslabel = None

    def withdraw(username, password):
        withdrawamount = int(withdrawEntry.get())
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="password", database="bankdb")

        cursor = mydb.cursor()
        selectquery = "UPDATE users SET accamount = accamount - %s WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery, (withdrawamount, username, password))

        selectquery2 = "SELECT accamount FROM users WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery2, (username, password))
        results = cursor.fetchall()

        global withdrawsuccesslabel
        withdrawsuccesslabel = Label(thirdpage, text="Amount Rs.%d has been debited " % (withdrawamount), font=("Times_New_Roman", 12), fg="red")
        withdrawsuccesslabel.place(x=250, y=250)

        mydb.commit()
        cursor.close()
        mydb.close()

    def cashDeposit(username, password):
        depositamount = int(cashdepositEntry.get())
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="password", database="bankdb")

        cursor = mydb.cursor()

        selectquery = "UPDATE users SET accamount = accamount + %s WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery, (depositamount, username, password))

        selectquery2 = "SELECT accamount FROM users WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery2, (username, password))

        results = cursor.fetchall()

        global depositsuccesslabel
        if withdrawsuccesslabel and withdrawsuccesslabel.winfo_exists():
            withdrawsuccesslabel.destroy()

        depositsuccesslabel = Label(thirdpage, text="Amount Rs.%d has been credited " % (depositamount), font=("Times_New_Roman", 12), fg="green")
        depositsuccesslabel.place(x=250, y=250)

        mydb.commit()
        cursor.close()
        mydb.close()

    def checkBalance(username, password):
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="password", database="bankdb")

        cursor = mydb.cursor()
        selectquery = "SELECT accamount FROM users WHERE username = %s AND passwd = %s"
        cursor.execute(selectquery, (username, password))
        results = cursor.fetchall()

        global balancesuccesslabel

        if depositsuccesslabel and depositsuccesslabel.winfo_exists():
            depositsuccesslabel.destroy()
        if withdrawsuccesslabel and withdrawsuccesslabel.winfo_exists():
            withdrawsuccesslabel.destroy()

        for i in results:
            balancesuccesslabel = Label(thirdpage, text="Your balance is Rs.%d " % int(i[0]), font=("Times_New_Roman", 12))
            balancesuccesslabel.place(x=250, y=250)

        cursor.close()
        mydb.close()

    thirdpage.title("VIT STUDENT BANK")
    thirdpage.geometry("700x500+200+150")

    withdrawlabel = Label(thirdpage, text="Withdraw : ", width=20)
    withdrawlabel.place(x=10, y=30)

    withdrawEntry = Entry(thirdpage, font=("Times_New_Roman", 12))
    withdrawEntry.place(x=200, y=30)

    withdrawButton = Button(thirdpage, text="Withdraw", width=20, command=lambda: withdraw(username, password))
    withdrawButton.place(x=460, y=30)

    depositlabel = Label(thirdpage, text="Deposit : ", width=20)
    depositlabel.place(x=10, y=70)

    cashdepositEntry = Entry(thirdpage, font=("Times_New_Roman", 12))
    cashdepositEntry.place(x=200, y=70)

    depositButton = Button(thirdpage, text="Deposit", width=20, command=lambda: cashDeposit(username, password))
    depositButton.place(x=460, y=70)

    balanceButton = Button(thirdpage, text="Check Balance", width=20, command=lambda: checkBalance(username, password))
    balanceButton.place(x=460, y=110)

def page3(user, pwd):
    global username
    global password
    username = user
    password = pwd
    thirdpage = Tk()
    screen3(thirdpage, username, password)
    thirdpage.mainloop()
