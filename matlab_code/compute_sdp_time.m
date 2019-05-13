clear
addpath('functions')

file_name = '../mat/20complete';
load_name = sprintf('%s_dataset',file_name);
save_name = sprintf('%s_time_sdp',file_name);
load(load_name)

n = size(A,1);
A2 = zeros(n+1,n+1,length(cw_range),nSample);
time_sdp = zeros(length(cw_range),nSample);



for cw = 1:length(cw_range)
for iter = 1:nSample
    A2(:,:,cw,iter) = [A(:,:,cw,iter),h(:,cw,iter);h(:,cw,iter)',0];
end    
end

len = (length(cw_range)*nSample);
maxNumCompThreads(1)

for i = 0:len-1
    iter = mod(i,nSample) + 1;
    cw = floor(i/nSample) + 1;
    tic
    [~,~,~] = mintraceNSD(A2(:,:,cw,iter));
    time_sdp(cw,iter) = toc;
    fprintf('cw=%f, iter=%d\n',cw_range(cw),iter);
end
save(save_name,'time_sdp')
