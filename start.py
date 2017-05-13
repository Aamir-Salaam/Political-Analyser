import subprocess
import os
import time
import csv
import matplotlib.pyplot as plt


import Aam_Aadmi_Party_data
import aap
import Bharatiya_Janata_Party_data
import bjp
import Indian_National_Congress_data
import inc


hour = time.strftime("%H")
minute = time.strftime("%M")
second = time.strftime("%S")
day = time.strftime("%d")
month = time.strftime("%m")
year = time.strftime("%Y")


f1 = open("aap.txt", "a");
value1 = ""+day+","+month+","+year+","+hour+","+minute+","+second+","+str(aap.ptweet)+","+str(aap.ntweet)+","+str(aap.netweet)+","+str(aap.pos)+","+str(aap.neg)+"\n"
myString1 = str(value1)
f1.write(myString1)
f1.close()

f2 = open("bjp.txt", "a");
value2 = ""+day+","+month+","+year+","+hour+","+minute+","+second+","+str(bjp.ptweet)+","+str(bjp.ntweet)+","+str(bjp.netweet)+","+str(bjp.pos)+","+str(bjp.neg)+"\n"
myString2 = str(value2)
f2.write(myString2)
f2.close()

f3 = open("inc.txt", "a");
value3 = ""+day+","+month+","+year+","+hour+","+minute+","+second+","+str(inc.ptweet)+","+str(inc.ntweet)+","+str(inc.netweet)+","+str(inc.pos)+","+str(inc.neg)+"\n"
myString3 = str(value3)
f3.write(myString3)
f3.close()


'''

x = [1,2,3]
y = [aap.ptweet,aap.netweet,aap.ntweet]
plt.title('AAM AADMI PARTY')
plt.ylabel("Tweet Count")
plt.xlabel("Sentiment")
plt.xticks([1,2,3],['Positive','Neutral','Negetive'])
plt.bar(x, y,width = 0.9, color = 'blue', edgecolor = 'black', linewidth = 2 ,align = 'edge')
plt.show()

'''

t =0
date = []
time =[]
pos = []
neg = []
temp = []



file = open('aap.txt', 'r')

for line in file.readlines():

    fname = line.rstrip().split(',') #using rstrip to remove the \n
    date.append(fname[0]+"/"+fname[1])
    time.append(fname[3]+":"+fname[4]+":"+fname[5])
    pos.append(float(fname[9]))
    neg.append(float(fname[10]))
    temp.append(int(t))
    t = t +1


plt.title('AAM AADMI PARTY')
plt.ylabel("Percentage")
plt.xlabel("Date")
plt.plot(temp,pos,color='GREEN',label='Positive')
plt.plot(temp,pos, 'ro')
plt.plot(temp,neg,color='RED',label='Negative')
plt.plot(temp,neg, 'ro')
plt.xticks(temp,date)
plt.legend()
plt.show()

t =0
date = []
time =[]
pos = []
neg = []
temp = []


file = open('bjp.txt', 'r')

for line in file.readlines():

    fname = line.rstrip().split(',') #using rstrip to remove the \n
    date.append(fname[0]+"/"+fname[1])
    time.append(fname[3]+":"+fname[4]+":"+fname[5])
    pos.append(float(fname[9]))
    neg.append(float(fname[10]))
    temp.append(int(t))
    t = t +1


plt.title('BHARTIYA JANTA PARTY')
plt.ylabel("Percentage")
plt.xlabel("Date")
plt.plot(temp,pos,color='GREEN',label='Positive')
plt.plot(temp,pos, 'ro')
plt.plot(temp,neg,color='RED',label='Negative')
plt.plot(temp,neg, 'ro')
plt.xticks(temp,date)
plt.legend()
plt.show()

t =0
date = []
time =[]
pos = []
neg = []
temp = []

file = open('inc.txt', 'r')

for line in file.readlines():

    fname = line.rstrip().split(',') #using rstrip to remove the \n
    date.append(fname[0]+"/"+fname[1])
    time.append(fname[3]+":"+fname[4]+":"+fname[5])
    pos.append(float(fname[9]))
    neg.append(float(fname[10]))
    temp.append(int(t))
    t = t +1


plt.title('INDIAN NATIONAL CONGRESS ')
plt.ylabel("Percentage")
plt.xlabel("Date")
plt.plot(temp,pos,color='GREEN',label='Positive')
plt.plot(temp,pos, 'ro')
plt.plot(temp,neg,color='RED',label='Negative')
plt.plot(temp,neg, 'ro')
plt.xticks(temp,date)
plt.legend()
plt.show()
