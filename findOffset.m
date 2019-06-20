function [offset] = findOffset(I)
    [G, E] = computeEntropyLoss(I);
    [~, offset] = min(E);
    offset = offset  + G(1);
end
    