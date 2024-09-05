% Ensayos cambiando las ganancias y h
%load('../logs/ensayo_real_1.mat')
load('../logs/ensayo_real_2.mat'); tmax=180; %tmax=300; % Sin Adaptacion
%load('../logs/ensayo_real_3.mat'); tmax=180; %tmax=190; % Adaptacion OK
%load('../logs/ensayo_real_4.mat'); tmax=170; % Adaptacion No OK

time = (reshape(time,1,length(time)));
base_time = 0;
normalized_time = zeros(1, length(time));

for i = 2:length(time)
    if (time(i) < time(i-1))
        base_time = base_time + 1;
    end
    normalized_time(i) = base_time + time(i)/1000000; 
end
dt = diff(normalized_time)/10;
Kd = Kdpi(:,1);
Kp = Kdpi(:,2);
Ki = Kdpi(:,3);

exec_time = reshape(exec_time,1,length(exec_time))/1e6;
mean_exec_time = mean(exec_time);
max_exec_time = max(exec_time);
min_exec_time = min(exec_time);
u = exec_time(1:end-1)./dt;

figure()
subplot(2,1,1)
plot(normalized_time, roll(:,1)-mean(roll(:,1)))
%legend('roll1','roll2')
xlabel('Time (s)')
ylabel('Roll Angle (rad)')
title(['Disturbance rejection - Mean Exec. Time: ', num2str(mean_exec_time*1e6,'%0.2f'), ' \mus'])
ylim([-6,6])
xlim([0,tmax])
grid on

subplot(2,1,2)
yyaxis left
plot(normalized_time(1:end-1), dt)
ylabel('Sampling Time (s)')
xlabel('Time (s)')
ylim([0,0.02])
xlim([0,tmax])
yyaxis right
plot(normalized_time(1:end-1), u)
ylabel('Utilization')
ylim([0,0.2])
%legend('Sample','Utilization')
grid on

% subplot(3,1,3)
% plot(normalized_time, [Kp,Ki,Kd])
% legend('Kp','Ki','Kd')
% xlabel('Time (s)')
% ylabel('Control Gains')
% ylim([0,1.5])
% xlim([0,tmax])
% grid on