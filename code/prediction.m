function [Pred, Error] = prediction(W, Y, C)
% input:
%   W: estimated W
%   Y: design matrix (without intercept) 
%   C: true label
n = size(Y, 1);
n_c = size(C, 2);
[~, Class] = max(C, [], 2);
if size(W, 2) == 1   
   W = reshape(W, [], n_c);
   if size(W,1)~=size(Y,2)
         Y = [Y ones(size(Y,1), 1)];
   end
   pred = Y*W;
   [~,predClass] = max(pred, [], 2);
   Error = sum(predClass ~= Class)/n;
else
   Error = zeros(0,size(W,2));
   if size(W,1)~=size(Y,2)
         Y = [Y ones(size(Y,1), 1)];
   end
   for i = 1:size(W,2)
       Wi = reshape(W(:,i), [], n_c);
       pred = Y*Wi;
       [~,predClass] = max(pred, [], 2);
       Error(i) = sum(predClass ~= Class)/n;
   end    
end 
Pred = predClass;
end