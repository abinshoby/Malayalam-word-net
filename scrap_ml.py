# -*- coding: utf-8 -*-
from selenium import webdriver
import re
import sqlite3
from sqlite3 import Error
import unicodedata
import os
import csv
def inc_count(count):
    with open("count.txt","w") as fp:
        fp.write(str(count))
    fp.close()

def get_last_count():
    with open("count.txt","r")as fp:
        count=int(fp.read())
    fp.close()
    return count


def add_to_csv(l):
   with open("test2.csv","a") as fp:
      a=csv.writer(fp,delimiter=',')
      a.writerows(l)

def scrap_word_syn(word):#find synonyms of a word
   path=str(os.popen("pwd").read()).strip("\n")+"/geckodriver"
   driver = webdriver.Firefox(executable_path=path)#/usr/local/bin/geckodriver
   word=word.encode('utf-8').decode('utf-8')
   driver.get("http://www.cfilt.iitb.ac.in/indowordnet/first?langno=9&queryword="+word)
   try:
       html=driver.find_element_by_id("words").get_attribute("innerHTML")
       l=re.findall(r'>(.*)<',html)
       org=[]
       for w in l:
         org.append(w.replace('\u200c', '').replace('\u200d', ''))
       driver.close()
       return org
   except  Exception as e:
      driver.close()
      return -1


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
   count=get_last_count()
   cur.execute("SELECT word FROM words_ml where _id>"+str(count))
   rows = cur.fetchall()
   return rows
conn=create_connection("enml.db")
if(conn):
   rows=select_all_tasks(conn)
   #print(scrap_word_syn("നിഘണ്ടു"))
   fin=[]
   c=get_last_count()      #currently the no of words is limited to 3 you can remove this condition for all words
   tot=0
   for row in rows:
         st=str(row[0])
         #print(st,"\n")
         q=st.split()
         if(tot>=6):
              break
         for i in q:
               k=scrap_word_syn(i)
               if(k!=-1):
                  k.insert(0,i)
                  fin.append(k)
                  add_to_csv(fin)
                  fin = []
                  #no=no+1
                  tot+=1
                  print(tot)
                  #print(i, ":")
                  #print(k)
               else:
                  continue
         c+=1
         inc_count(c)





