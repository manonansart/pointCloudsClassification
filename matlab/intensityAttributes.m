
% Expected path format : 'foo/'

function attributes = intensityAttributes(fpath)
	meanIntensity = [];
	maxIntensity = [];
	varIntensity = [];

	% Loop over all pcd files in fpath
	files = dir(strcat(fpath, '*.pcd'));
	for file = files'
	    data = loadpcd(strcat(fpath, file.name));

	    intensity = data(4, :)';

		meanIntensity = [meanIntensity ; mean(intensity)];
		maxIntensity = [maxIntensity ; max(intensity)];
		varIntensity = [varIntensity ; var(intensity)];
	end

	attributes = [meanIntensity maxIntensity varIntensity];

	% Save the attributes to csv format
	csvwrite(strcat(fpath,'intensityAttributes.csv'), attributes);