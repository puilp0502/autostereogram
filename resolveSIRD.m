function [dmap] = resolveSIRD(I)
dmap = computeDiff(I);
dmap(dmap < 0) = 0;