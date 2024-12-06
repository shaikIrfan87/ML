import csv
a=[]
with open('1-document.csv','r')as csvfile:
    next(csvfile)
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)
print("\n The total no. of training instances are:",len(a))
num_attribute=len(a[0])-1
print("\n the initial hypothesis is:")
hypothesis=['0']*num_attribute
print(hypothesis)
for i in range (0,len(a)):
    if a[i][num_attribute]=='yes':
        print("\ninstance",i+1,"is",a[i],"and is positive instance")
        for j in range(0,num_attribute):
            if hypothesis[j]=='0' or hypothesis[j]==a[i][j]:
                hypothesis[j]=a[i][j]
            else:
                hypothesis[j]='?'
        print("the hypothesis for the training instance",i+1,"is:",hypothesis,"\n")
    if a[i][num_attribute]=='no':
        print("\n Instance",i+1,"is",a[i],"and is negative instance .hence ignored")
        print("The hypothesis specific hypothesis for the training instance",i+1,"is:",hypothesis,"\n")
print("\n The normally specific hypothesis for the training instance is",hypothesis)       
    
