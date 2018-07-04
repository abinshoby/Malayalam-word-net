# -*- coding: utf-8 -*-
from selenium import webdriver
import re
import sqlite3
from sqlite3 import Error
import unicodedata
import os
import csv
def add_to_csv(l):
   with open("test.csv","w") as fp:
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
   cur.execute("SELECT word FROM words_ml")
   rows = cur.fetchall()
   return rows
conn=create_connection("enml.db")
if(conn):
   rows=select_all_tasks(conn)
   #print(scrap_word_syn("നിഘണ്ടു"))
   fin=[]
   no=0  #currently the no of words is limited to 3 you can remove this condition for all words
   for row in rows:
         st=str(row[0])
         #print(st,"\n")
         q=st.split()
         if(no>=3):
            break
         for i in q:
               k=scrap_word_syn(i)
               if(k!=-1):
                  k.insert(0,i)
                  fin.append(k)
                  no=no+1
                  #print(i, ":")
                  #print(k)
               else:
                  continue

   add_to_csv(fin)


