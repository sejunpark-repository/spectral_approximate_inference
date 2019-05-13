function [U,S] = modeig(H)
if ~isequal(H,H')
    fprintf('modeig: input matrix is not symmetric\n')
    return
end
n = size(H,1);

[U,S] = eig(H);
% for i = 1:n
%     if U(:,i)' * V(:,i) < 0
%         S(i,i) = -S(i,i);
%     end
% end
[S,I] = sort(diag(S),'descend');
S = diag(S);
U = U(:,I);
% H2 = U*S*U';
% a=0;
end

