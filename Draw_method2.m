v = readtable('all_point.csv');

CC = readtable('Circels.csv');

v=v{:,:}
CC=CC{:,:}



X=CC(:,1)
Y=CC(:,2)
r=50
circles (X,Y,r)


XX=v(:,1)
YY=v(:,2)

hold on 
plot (XX,YY,'o')

for j=2:length(v)-1
    i=j
plot ([XX(i) XX(i+1)], [YY(i) YY(i+1)],'k')
end

axis([0 2483 0 2483/2])


 


