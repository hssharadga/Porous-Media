
v = readtable('all_point.csv');

CCC = readtable('sphere_center.csv');
p = readtable('x.csv');

p=p{:,:}
v=v{:,:}
CCC=CCC{:,:}
X=CCC(2:end,1)
Y=CCC(2:end,2)
r=50
figure
circles (X,Y,r)


XX=v(1:end,1)
YY=v(1:end,2)

% XX=p(1:end,1)
% YY=p(1:end,2)


% XX=kk(2:end,1)
% YY=kk(2:end,2)

hold on 
plot (XX,YY,'o')

for j=1:length(v)-1
    i=j
p1 = [XX(i) YY(i)];                         % First Point
p2 = [XX(i+1) YY(i+1)];                         % Second Point
dp = p2-p1;                         % Difference
quiver(p1(1),p1(2),dp(1),dp(2),'k','AutoScale','off','MaxHeadSize',0.5,'linewidth',1)


end

axis([-2483 2483 -100 2483/2])


% plot ((p(2:end,1)),(p(2:end,2)),'k*')
 
% circles (jk(:,1),jk(:,2),r)

