#from cusat using api
# -*- coding: utf-8 -*-
import requests
import csv
import sqlite3
import json
from sqlite3 import Error
import unicodedata
def add_to_csv(l):
    f = open('test6_cusat.csv',"a")
    csv_file = csv.writer(f)
    l = json.loads(l)
    csv_file.writerow([str(l[0]['sid']), l[0]['synset'] ,l[0]['meaning'] , l[0]['pos']] + l[0]['hypernym'] + l[0]['hyponym'] + l[0]['meronym'] + l[0]['antonym'])
    f.close()


def scrap_word_syn(word):
    #url = "http://malayalamwordnet.cusat.ac.in/restapi/json/synset?searchWord="+word
    url = "http://malayalamwordnet.cusat.ac.in/restapi/json/synset/"+str(word)
    response = requests.request("GET", url)
    if(response.status_code==200):
        print(response.text)
        return response.text
    else:
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
   #count=get_last_count()
   cur.execute("SELECT word FROM words_ml where _id;")
   rows = cur.fetchall()
   return rows
conn=create_connection("enml.db")
if(conn):
   rows=select_all_tasks(conn)
   #print(scrap_word_syn("നിഘണ്ടു"))
   fin=[]
   #c=get_last_count()      #currently the no of words is limited to 3 you can remove this condition for all words
   tot=0
   #for row in rows:
         #st=str(row[0])
         #print(st,"\n")
        # q=st.split()
        # if(tot>=6):
         #     break
         #for i in q:
   i=20028

   while(True):
               k=scrap_word_syn(i)
               i+=1
               if(k!=-1 and len(k)>0):
                    add_to_csv(k)
                    tot+=1
                    print(tot)





         #c+=1
        # inc_count(c)