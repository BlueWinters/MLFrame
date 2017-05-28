function acc = cnnCrossValidation(cnn, vx, vy)
vmid = cnnForwardPropagate(cnn, vx, vy);
[~, prelabels] = max(vmid.fMaps{end});
[~, labels] = max(vy);
acc = mean(labels(:) == prelabels(:));
end