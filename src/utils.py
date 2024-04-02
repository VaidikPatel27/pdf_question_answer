import os

def save_data(path, file, type = 'wb'):
	if os.path.exists(path):
		os.remove(path)
		with open(path, type) as f:
			f.write(path, type)
	else:
		with open(path, type) as f:
			f.write(path, type)

