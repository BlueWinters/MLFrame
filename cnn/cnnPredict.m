function acc = cnnPredict(cnn, data)
mid = cnnForwardPropagate(cnn, data.x, data.y);
[~, prelabels] = max(mid.fMaps{end});
[~, labels] = max(data.y);
acc = mean(labels(:) == prelabels(:));
end