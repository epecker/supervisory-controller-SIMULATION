% Ensayos cambiando solo las ganancias (sin modificar h)
%load('../logs/ensayo_kd0p3_ki1_ki0p3.mat')
%load('../logs/ensayo_kd0p15_ki1_ki0p6.mat')
load('../logs/ensayo_kd0p6_ki1_ki0p15.mat')

Ts = 5e-3;
Kd = 0.3;
Kp = 1;
Ki = 0.3;

Kd_exp = Kdpi(end,1);
Ki_exp = Kdpi(end,3);
N = Ki/Ki_exp;

time = (reshape(time,1,length(time)));

base_time = 0;
normalized_time = zeros(1, length(time));

for i = 2:length(time)
    if (time(i) < time(i-1))
        base_time = base_time + 1;
    end
    normalized_time(i) = base_time + time(i)/1000000; 
end

figure()
plot(normalized_time, roll)
legend('roll1','roll2')
xlabel('Time (s)')
ylabel('Roll Angle (rad)')
title(['Disturbance rejection - h_{\phi} = ',num2str(Ts,'%0.1e'),' ms, Kp = ',num2str(Kp), ', Ki = ', num2str(Ki_exp), ', Kd = ', num2str(Kd_exp)])
ylim([-10,20])
grid on
