
v = readtable('all_point.csv');

CCC = readtable('sphere_center.csv');
v=v{:,:}
CCC=CCC{:,:}
X=CCC(2:end,1)
Y=CCC(2:end,2)
r=50
circles (X,Y,r)


XX=v(2:end,1)
YY=v(2:end,2)

hold on 
plot (XX,YY,'o')

for j=1:length(v)-2
    i=j
plot ([XX(i) XX(i+1)], [YY(i) YY(i+1)],'k')
end

axis([-2483 2483 -100 2483/2])


 


