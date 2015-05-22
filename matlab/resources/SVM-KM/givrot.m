function [G] = givrot (x);
%GIVROT
%   [G] = givrot (x);
%   returns Givens-rotation matrix
%   [c -s; s c] which rotates x(2) to zero.

  if (x(2) ~= 0),
    r = norm (x, 2);
    G = 1/r*[x(1) x(2); -x(2) x(1)];
  else
    G = eye (2);
  end


