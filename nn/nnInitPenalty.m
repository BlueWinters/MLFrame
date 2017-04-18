function nn = nnInitPenalty(nn)
nn.penaltyVal{l}.pCost = 0;
nn.penaltyVal{l}.pDelta = 0;
nn.penaltyVal{l}.pWdiff = 0;
nn.penaltyVal{l}.pbDiff = 0;
end