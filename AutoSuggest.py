# -*- coding: utf-8 -*-
from tkinter import *
import re
import sqlite3
from sqlite3 import Error
from predict_text_meaning import *


def create_connection(db_file):
   """ create a database connection to the SQLite database
       specified by the db_file
   """
   try:
      conn = sqlite3.connect(db_file)
      return conn
   except Error as e:
      print(e)

   return None


def select_all_tasks(conn):
   """
   Query all rows in the tasks table
   """
   cur = conn.cursor()
   cur.execute("select word from test union select word from test;")
   rows = cur.fetchall()
   return rows

class AutocompleteEntry(Entry):
    def __init__(self, lista, *args, **kwargs):

        Entry.__init__(self, *args, **kwargs)
        self.lista = lista
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)

        self.lb_up = False

    def changed(self, name, index, mode):

        if self.var.get() == '':
            self.lb.destroy()
            self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = Listbox(width=50)
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.place(x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                    self.lb_up = True

                self.lb.delete(0, END)

                for w in words:
                    self.lb.insert(END, w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False

    def selection(self, event):

        if self.lb_up:
            self.var.set(self.lb.get(ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(END)

    def up(self, event):

        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index) - 1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def down(self, event):

        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != END:
                self.lb.selection_clear(first=index)
                index = str(int(index) + 1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        #1pattern = re.compile('.*' + self.var.get() + '.*')
        #1return [w for w in self.lista if re.match(pattern, w)]
        #patt1=re.compile(self.var.get()+'.*') #correct1
        #return [w for w in self.lista if re.match(patt1,w)] #correct1
        out=predict(self.var.get().strip())
        return out


if __name__ == '__main__':
    conn=create_connection("syn.db")
    if(conn):
        rows=select_all_tasks(conn)
        lista=[]
        for tup in rows:
            #print(tup[0])
            lista.append(str(tup[0]).replace('\u200c', '').replace('\u200d', ''))
        print(lista)
    else:
        print("db error")
    root = Tk()
    root.minsize(1000,500)
    entry = AutocompleteEntry(lista, root, font=('Lohit Malayalam ', 15))
    entry.grid(row=0, column=0)
    entry.config(width=50)
    entry.pack()
    #Button(text='nothing',height=4,width=100).grid(row=1, column=0)
    #Button(text='nothing',height=4,width=100).grid(row=2, column=0)
    #Button(text='nothing',height=4,width=100).grid(row=3, column=0)

    root.mainloop()