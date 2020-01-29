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
% plot (XX,YY,'o')

for j=2:length(v)-1
    i=j
p1 = [XX(i) YY(i)];                         % First Point
p2 = [XX(i+1) YY(i+1)];                         % Second Point
dp = p2-p1;                         % Difference
quiver(p1(1),p1(2),dp(1),dp(2),'k','AutoScale','off','MaxHeadSize',0.5,'linewidth',1)


end

axis([0 2483 0 2483/2])


 


