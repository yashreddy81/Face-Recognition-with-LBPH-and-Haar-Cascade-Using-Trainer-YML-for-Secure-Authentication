import mysql.connector

def connect_db():
    mydb = mysql.connector.connect(host="localhost", user="root", passwd="password", database="bankdb")
    return mydb

def create_user(username, password):
    mydb = connect_db()
    cursor = mydb.cursor()
    cursor.execute("INSERT INTO users (username, passwd, accamount) VALUES (%s, %s, %s)", (username, password, 0))
    mydb.commit()
    cursor.close()
    mydb.close()

def check_user(username, password):
    mydb = connect_db()
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s AND passwd = %s", (username, password))
    result = cursor.fetchone()
    cursor.close()
    mydb.close()
    return result
