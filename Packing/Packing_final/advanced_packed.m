
clc
clear all
CC = readtable('Circels.csv');
CC1 = readtable('Center_old.csv');

for i=1:2


CCC=CC{:,:}
CCC1=CC1{:,:}

if i==1   
X=CCC(:,1)
Y=CCC(:,2)
r=50
figure
circles (X,Y,r)
end
i=3
if i==2
XX=CCC1(:,1)
YY=CCC1(:,2)
r=50
figure
circles (XX,YY,r)  
end

axis([0 2483*2 0 2483/2])

end

 


