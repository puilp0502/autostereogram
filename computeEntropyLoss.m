function [G, E] = computeEntropyLoss(I)
    computeFrom = floor(size(I, 2) / 24);
    computeTo = floor(size(I, 2) / 4);
    E = zeros(1, computeTo - computeFrom + 1);
    G = computeFrom:computeTo;
    i = 1;
    for gap = G
        dmap = computeDiff(I,gap);
        failed = sum(dmap < 0, 'all') / numel(dmap) * 100;
        E(i) = failed;
        i = i + 1;
    end
end