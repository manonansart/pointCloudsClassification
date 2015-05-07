close all
clear all


labels_track = load('../../dataset/lomita/labels_track.csv');
labels = [];

for i = 0:154
	if (i < 10)
	num_chaine = strcat('00', int2str(i));
	% Loop over all pcd files in fpath
	elseif (i < 100)
		num_chaine = strcat('0', int2str(i));
	else
		num_chaine = int2str(i);
	end
	files = dir(strcat('../../dataset/lomita/track_', num_chaine, '*.pcd'));
	for file = files'
		labels = [ labels; labels_track(i + 1)];
	end
end

csvwrite('../../dataset/lomita/labels.csv', labels)