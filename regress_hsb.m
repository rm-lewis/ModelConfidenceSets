function [b,tStat] = regress_hsb(y,X)

% Performs a linear regression of y on X. 

% Input:
% - y (response vector): a vector
% - X (design matrix): a matrix with the same number of rows as y

% Output:
% - b (the ordinary least squares estimate): a vector with p rows, where p
%   is the number of columns of X
% - tStat (the t-statistic in a linear regression): a vector with p rows,
%   where p is the number of columns of X

[n,ncolX] = size(X);
[Q,R,perm] = qr(X,0);
if isempty(R)
    p = 0;
elseif isvector(R)
    p = double(abs(R(1))>0);
else
    p = sum(abs(diag(R)) > max(n,ncolX)*eps(R(1)));
end
if p < ncolX
    R = R(1:p,1:p);
    Q = Q(:,1:p);
    perm = perm(1:p);
end
b = zeros(ncolX,1);
b(perm) = R \ (Q'*y);
    RI = R\eye(p);
    nu = max(0,n-p);                % Residual degrees of freedom
    yhat = X*b;                     % Predicted responses at each data point.
    r = y-yhat;                     % Residuals.
    normr = norm(r);
    if nu ~= 0
        rmse = normr/sqrt(nu);      % Root mean square error.
    else
        rmse = NaN;
    end
    s2 = rmse^2;                    % Estimator of error variance.
    se = zeros(ncolX,1);
    se(perm,:) = rmse*sqrt(sum(abs(RI).^2,2));
    tStat=b./se;