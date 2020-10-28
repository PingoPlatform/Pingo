def convert_world_file_to_pixel(utm_coord, world_file_name):
	#Nimmt referenzierte Koordinaten, und berechnet Pixel-Koordinaten mit Hilfe des World-Files
	#Structure of a world file:
	#Line 1: A: pixel size in the x-direction in map units/pixel
	#Line 2: D: rotation about y-axis
	#Line 3: B: rotation about x-axis
	#Line 4: E: pixel size in the y-direction in map units, almost always negative[3]
	#Line 5: C: x-coordinate of the center of the upper left pixel
	#Line 6: F: y-coordinate of the center of the upper left pixel

	out=[]
	temp=[]
	
	#Auslesen der Parameter aus dem World File
	try:
		f = open(world_file_name,'r')
		print("Lese aus Worldfile: " , world_file_name)
		for line in f:
			temp.append(line.strip())
		pixel_size_x=float(temp[0])
		pixel_size_y=float(temp[3])
		upper_left_x=float(temp[4])
		upper_left_y=float(temp[5])

	except:
		print("Fehler beim Einlesen des World-Files!")
		return
	
	#Umrechnen der Parameter auf Pixel: Hier ist eine Ungenauigkeit drin, Nachkommastellen werden am Ende ignoriert.
	
	for i in range(len(utm_coord)):
		pixel_y = (upper_left_y - utm_coord[i][0]) / pixel_size_y
		pixel_y = abs(int(pixel_y))
		pixel_x = (utm_coord[i][1] - upper_left_x) / pixel_size_x
		pixel_x = abs(int(pixel_x))
		out.append((pixel_y, pixel_x))

	return out
	
def convert_pixel_to_world_file(pixel_coord, world_file_name):

	out=[]
	temp=[]
	
	#Auslesen der Parameter aus dem World File
	try:
		f = open(world_file_name,'r')
		print("Lese aus Worldfile: " , world_file_name)
		for line in f:
			temp.append(line.strip())
		pixel_size_x=float(temp[0])
		pixel_size_y=float(temp[3])
		upper_left_x=float(temp[4])
		upper_left_y=float(temp[5])

	except:
		print("Fehler beim Einlesen des World-Files!")
		return
		
	#Berechnen der Koodinaten aus den Pixel
	for i in range(len(pixel_coord)):
		utm_y = upper_left_y + (pixel_coord[i][0] * pixel_size_y)
		utm_x = upper_left_x + (pixel_coord[i][1] * pixel_size_x)
		out.append((utm_y, utm_x))
	
	return out


def utm_convert(df, X = 'X', Y = 'Y'): 
    import utm

    #Takes a pandas dataframe with latlong coordinated and adds the column UTX and UTMY
    # Problems with mutlidimensional arrays. alternative:
    '''
    import pyproj
    from pyproj import Proj
    myProj = Proj("+proj=utm +zone=33U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    df['utmx'],df['utmy'] = myProj(lon_array, lat_array) 
    '''
    print('Calculate X')
    df['UTMX'] = df.apply(
        lambda x: utm.from_latlon(x['Y'], x['X'])[0], axis = 1 )
    print('Calculate Y')
    df['UTMY'] = df.apply(
        lambda x: utm.from_latlon(x['Y'], x['X'])[1], axis = 1 )
    
    return df

def utm_convert_to_lalo(df, Zone = 32,  X = 'X', Y = 'Y'): 
    import utm

    #Takes a pandas dataframe with latlong coordinated and adds the column UTX and UTMY
    print('Calculate X')
    df['LaLo_X'] = df.apply(
        lambda x: utm.to_latlon(x['Y'], x['X'])[0], Zone, 'U', axis = 1 )
    print('Calculate Y')
    df['LaLo_Y'] = df.apply(
        lambda x: utm.to_latlon(x['Y'], x['X'])[1], axis = 1 )
    
    return df