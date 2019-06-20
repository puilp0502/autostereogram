function [disparityMap] = computeDiff(I, varargin)
%computeDiff - compute depth map of autostereogram using block matching disparity calculation
%  The rational behind ths function is that the autostereogram image is
%  essentially "merged" stereographic image. By splitting an image into left-
%  eye's perspective and right eye's perspective, we can compute the depth 
%  map used to create the autostereogram.
%  Parameters:
%    I: autostereographic image
%    gap: base pattern repetition interval

% check argument number
if (nargin > 2)
    error("computeDiff accepts at most 2 arguments");
end
shape = size(I);

% If image is RGB, convert to greyscale image
if (length(shape) == 3)
    I = rgb2gray(I);
end

% if offset is not given, try to find the "best" offset
if (nargin == 1)
    offset = findOffset(I);
else
    offset = varargin{1};
end

% Calculate the left-eye perspective
left = I(:, 1:shape(2)-offset);
% Calculate the right-eye perspective
right = I(:, offset+1:end);

disparityRange = [0 16];
% Calculate the disparity map using block matching
disparityMap = disparity(left, right, 'BlockSize', 19, ...
    'DisparityRange', disparityRange, ...
    'Method', 'BlockMatching');
