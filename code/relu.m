function [y] = relu(x)
   n = numel(x);
   y = zeros(size(x));
   for i = 1:n
     if x(i) >= 0
        y(i) = x(i);
     end
   end