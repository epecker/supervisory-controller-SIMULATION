%battery=(reshape(battery,1,length(battery)));
%height=(reshape(height,1,length(height)));
time=(reshape(time,1,length(time)));

base_time=0;
normalized_time=zeros(1,length(time));

for i=2:length(time)
    if (time(i)<time(i-1))
        base_time=base_time+1;
    end
    normalized_time(i)=base_time+time(i)/1000000; 
end

clear base_time
clear i
clear tout