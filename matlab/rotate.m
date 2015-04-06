function newData = rotate(data)
	x = sum(data(:, 1));
	y = sum(data(:, 2));

	theta = atan((2 * x * y) / (x^2 - y^2)) / 2;

	R = [ cos(theta), -sin(theta), 0;
		sin(theta), cos(theta), 0;
		0, 0, 1];

	newData = data;
	newCoordinates = R * data(:, 1:3)';
	newData(:, 1:3) = newCoordinates';
end