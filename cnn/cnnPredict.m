function acc = cnnPredict(cnn, x, y)
mid = cnnFeedforward(cnn, x, y);
[~, prelabels] = max(mid.fMaps{end});
[~, labels] = max(y);
acc = mean(labels(:) == prelabels(:));
end