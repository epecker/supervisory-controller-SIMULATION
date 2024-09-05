% Ensayos cambiando las ganancias y h
T = readtable('../logs/Table1_overleaf.csv','NumHeaderLines',1);  % skips the first row of data
T = table2array(T);
[nc, nr] = size(T);
k = 1:nc;

hfbs   = T(:,1);
Jroll  = T(:,2);
Jpitch = T(:,3);
Jyaw   = T(:,4);
Jtotal = T(:,5);
Jqs    = T(:,6);
Jc     = T(:,7);
xlabels_table = {};
for i = k
    xlabels_table{end+1} = hfbs(i);
end

figure()
subplot(2,1,1)
plot(k, [Jroll,Jpitch,Jyaw,Jtotal],'-s',MarkerSize=6)
legend({'J_{c,roll}','J_{c,pitch}','J_{c,yaw}','J_{c,total}'},'Orientation','horizontal')
xlabel('h_{fbs} (s)')
ylabel('Attitude cost functions')
xticks(k)
xticklabels(xlabels_table)
title('Sensitivity analysis - Cost functions at T_{sim}')
grid on

subplot(2,1,2)
yyaxis left
plot(k, Jqs,'-s',MarkerSize=6)
ylabel('Queue-Server cost J_{qs}')
xlabel('h_{fbs} (s)')
yyaxis right
plot(k, Jc,'-s',MarkerSize=6)
ylabel('Total cost J_{c}')
xticks(k)
xticklabels(xlabels_table)
grid on
