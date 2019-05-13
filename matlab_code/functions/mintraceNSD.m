function [U,S,D] = mintraceNSD(A)
n = size(A,1);
if ~isequal(diag(A),zeros(n,1))
    fprintf('mintraceNSD: input has non-zero diagonal\n')
end

cvx_begin sdp quiet
   variable D(n,n) diagonal
   minimize(trace(D))
   D >= A   
cvx_end

[U,S] = modeig(A-D);

