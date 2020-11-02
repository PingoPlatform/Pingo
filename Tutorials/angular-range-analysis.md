---
description: 'Author: Peter Feldens'
---

# Angular Range Analysis

### Introcution to ARA analysis

Angular range analysis (ARA) utilizes the decrease of backscatter intensities with increasing angle of incidence on the seafloor. It is chiefly controlled by the relevant roughness of the seafloor in relation to the utilized acoustic wavelength, and in many shows patterns that are charactistic for different seafloor composition. 

### Loading the data

We use a 400 kHz file recorded offshore Rostock (Germany) in a shallow part of the Baltic Sea. No correction regarding erroneous data, patch test or draft are done for the purpose of this chapter. We will only use the single line, and not the complete mosaic, to speed up the processing. The procedure would work identically with the datalist of the complete mosaic, feel free to try that if you are interested. While the aim using mbbackangle was to get rid of angular relationships for the creation of backscatter mosaics, here we want to utilize the relationship for seafloor classification.

The file 20191017_090105_kh1910_400khz.mb89 used here can be downloaded ffrom the following folder: https://www.dropbox.com/sh/yfk0as0bw4790s1/AADdPWMtKRC-Yh06xXX8oTjsa?dl=0 . Copy the file in the ./data folder of this git folder. 

First, we do the preprocessing of the file to load it into mbsystem:

```text
mbpreprocess --format=89 --input=./data/20191017_091850_kh1910_400khz.s7k --multibeam-sidescan-source=S 
```

Note the file contains a field with calibrated snippet information (field 7058 in a Reson s7k files, the same fields are supported in other Software such as FMGT or SonarWiz during import), if we set the side scan source to "C". For this tutorial, we use the more commonyl available field 7028, which is imported with the switch "S".

Here, we do not do any further correction and export the data we require for the ARC curvces

```text
mblist -MA -OXYNCd#.lBg -F89 -I./data/20191017_091850_kh1910_400khz.s7k  > 20191017_091850_kh1910_400khz.arc
```

This mblist option exports the coordinates, the ping count, the transducer height, the across track distance, the beam number, the pulse lengths and the backscatter values to a file with the ending .arc. Note that to my knowledge it is not possible to directly export angles for snippet-derived sidescan data with mbsystem, nor is it possible to export the frequency for s7k information. This will complicate our postprocessing a bit. 

A direct plot of angles would be possible for the export of beam amplitude data (change b to B and -NA to -MA to achieve this).

This information includes everythng we require for the ARC analysis, and we switch to python for the further analysis. You could use a jupyter notebook or jupyter qtconsole to follow the code. We use the jupter qtconsole here. 

In the following, we will need to apply corrections for absorptiom static survey gains and spreading as well as a footprint correction that is not yet incorporated in the above s7k files (by specification, data in field 7058 should be footprint corrected, but this file is not).

Enable inline plotting and load the pandas and numpy dataframe module \(within the console\). Pandas is a powerful frame for quickly storing and manipulating data we will use, while numpy is a C-based module that allows very fast mathematical operations on arrays within Python.

```text
%pylab inline
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
```

Load the raw data we exported to the ASCII file into the pandas dataframe df\_raw.

```text
df_raw = pd.read_csv('test.arc', names = ['X', 'Y', 'Ping', 'Height', 'Across_Dist', 'Beam', 'PulseLength', 'BS', 'Angle'], sep = '\t')
```

Lets plot the first 1000 data points. The method .iloc references the rows of a pandas dataframe. Note that indices and array in Python are enclosed in \[ \], while parameters for functions are specified within \( \).

```text
df_raw.iloc[0:20000].plot.scatter(x='Angle', y='BS')
```
![](./img/arc_1.png)

Some variance in the curves is apparent as well as some outliers. We will not try to correct or remove bad pings at all, can have a noticeable effect later on. Normally, considerable effort should go into cleaning the multibeam data.

If interested: Follow the tutorial in the creation of backscatter mosaics, and plot the points for a datafile processed after mbbackangle was used. Name the dataframe df\_proc. Do we continue with the raw or processed data for angular response analysis?

### Parameterize the data and apply the sonar equation

Before we can use the ARA curves for plotting, the data needs to be parameterized. A suitable choice for that purpose is the GSAB model \(Lamarche et al., 2011; [https://www.researchgate.net/publication/222161889\_Quantitative\_characterisation\_of\_seafloor\_substrate\_and\_bedforms\_using\_advanced\_processing\_of\_multibeam\_backscatter-Application\_to\_Cook\_Strait\_New\_Zealand](https://www.researchgate.net/publication/222161889_Quantitative_characterisation_of_seafloor_substrate_and_bedforms_using_advanced_processing_of_multibeam_backscatter-Application_to_Cook_Strait_New_Zealand)\)

Since we already have the ARA curves in a pandas dataframe, application of the model is comparatively straightforward. The tasks involve fitting the raw data to a curve function given by the GSAB model and then calculating the parameters A to F. This needs to be done for every or a small number of pings, otherwise we would calculate the average over larger areas. Fortunately, we already exported a PingCount variable with mblist and loaded it into the dataframe. Also, we need to convert the linear intensity data to decibels.

For the next operations, we prepared some custom functions.
```text





```


Now we convert the linear amplitude to dB values, which we store in a new column of the dataframe. We get a lot of -inf values due to many 0-intensity data points. We will remove these points as well as missing datapoints from our dataset. Take a moment to understand the logic and syntax in the command.

```text
#Make new column in dataframe for dB values
df_raw['BS_dB'] = 20 * np.log10(df_raw['BS'])

#Remove the -infs caused by log(0) with 0
df_raw.drop(df_raw[df_raw.isin([np.nan, np.inf, -np.inf]).any(1)].index, inplace = True)
```

Calibrated high-frequency backscatter data typically shows intensities between -70 and 0 dB depending on angle and seafloor. Our data here is uncalibrated and should be corrected according to the sonar equation as far as possible. The \(incomplete\) sonar equation is given with: TS = EL - SL + 2\*TL with TS: target strength, EL: echo level, SL: source level, and TL: transmission loss. The transmission loss TL is:

TL = Spreading + Absorption

For brevity, we ignore the effects of the insonifed area; details for the computation could be found in \(Lamarche and Lurton, 2015\). Prior to computing the sonar equation, the gains applied during the survey have to be removed to obtain the echo level EL. Fortunately, for this survey everything was recorded without gain. The correct absorption values for the area and frequency is 90 dB/km.

We correct the dB intensity values in the dataframe. In addition, we calculate the required slant range based on transducer altitude and across-track range \(ignoring along-track components\) that we exported with mblist.

```python
spreading_survey = 0
absorption_survey = 0  # dB/km
static_survey = 0

# Get slant range in m
df_raw['Slant'] = np.sqrt(df_raw.Altitude**2 + df_raw.AcrossDist**2)

# Remove an eventual survey gain
df_raw['EL'] = eco.remove_survey_gain(spreading_survey, static_survey, absorption_survey, df_raw.Intensity_dB, df_raw.Slant)

# Apply the simplified Sonar equation
absorption_new = 90 #db/km
SL = 210   # This is estimated for the 200 kHz frequency
df_raw['TS'] = eco.sonar_equation(absorption_new, SL, df_raw.EL, df_raw.Slant)

#Plot the first 20000 data points
df_raw.iloc[0:20000].plot.scatter(x='Angle', y='TS')
```

Lets have a look at our column names and remove the ones we dont need to clean up the dataframe a bit.

```text
df_raw.columns
df_raw.drop(['Intensity','Intensity_dB', 'EL'], axis=1, inplace = True)
```

We have a very high angle resolution that we do not need and which will slow done the ARA analysis massively. Thus, lets bin the data into 2° and 30 ping intervals. In addition, we limit the data to incidence angles &gt; 0° \(thus looking only at the starboard part of the swath width\).

```python
#Group data by Ping Counts and bin into 2° intervals, stacking over 30 pings 
bins_angle = np.arange(-80,80,2) 
ping_bin_number = 30
bins_ping = np.arange(df_raw.PingCount.min()-ping_bin_number,df_raw.PingCount.max()  +   ping_bin_number,ping_bin_number) 

## Binning Angles
# Sort incidence angles indices into bins_angle
bins_angle_idx = np.digitize(df_raw.Angle, bins_angle)
# Make new dataframe columns
data = []  
for i in range(bins_angle_idx.size):
     data.append(bins_angle[bins_angle_idx[i]])   #Right boundary, not precise
df_raw['Angle_binned'] = data

## Binning Pings
bins_ping_idx = np.digitize(df_raw.PingCount, bins_ping)
data = []  
for i in range(bins_ping_idx.size):
     data.append(bins_ping[bins_ping_idx[i]])
df_raw['Pings_binned'] = data

# Make a new dataframe that includes the binned angles and pings
df_bins = df_raw.groupby([df_raw.Pings_binned, df_raw.Angle_binned], as_index=False).mean()

# Drop values with angles < 0 and > 60°
df_bins.drop(df_bins[(df_bins.Angle_binned < 0) | (df_bins.Angle_binned > 60)].index, inplace = True)

# Again, We drop lines with NaN's  and infs
df_bins.drop(df_bins[df_bins.isin([np.nan, np.inf, -np.inf]).any(1)].index, inplace = True)
```

We continue working with the reduced dataset df\_bins. Actual calculation of the GSAB involves: \#\#\# We split the datasets into the individual pings \#\#\# a small scale example is: demo = df\_bins\[0:200\] \# get the first pings demo\_pings = demo.groupby\('Pings\_binned'\) \# split in pings for demo\_ping\_number, demo\_ping\_data in demo\_pings: print\(demo\_ping\_number\) print\(demo\_data\_number\) \#\#\#

```python
# Fit GSAB model
import numpy as np
from scipy.optimize import curve_fit

def GSAB_func(theta, A, B, C, E, F):
     # Calculates the GSAB Function
        argument_1 = A*np.exp(-theta**2/(2*B**2))
        argument_2 = C*np.cos(theta)**2
        argument_3 = E * np.exp(-theta**2 / (2*F**2))
        BS = 10*np.log10(argument_1 + argument_2 + argument_3 )
        return BS

# Split data into individual pings
pings = df_bins.groupby('Pings_binned')

# set bounds for A-F to speed up fitting. Roughly based on values reported by Lamarche et al (2011).
bounds_min = [10**(-35/10), 0.1*np.pi/180, 10**(-30/10), 10**(-30/10), 5*np.pi/180]
bounds_max = [10**(25/10), 35*np.pi/180, 10**(5/10), 10**(20/10), 25*np.pi/180]

# Iterate over individual pings. This could be vectorized and be much faster for 
# application on real datasets. The loop may take a moment.
results = []
for ping, data in pings:
    theta = data.Angle_binned * np.pi/180   # get angles to rad
    #Fit paramameters to the GSAB function using scipys curve_fit function. 
    popt, pcov = curve_fit(GSAB_func, theta, data.TS, bounds=(bounds_min, bounds_max))
    # Calculate the parameters A-F. D is fixed at 2. 
    A = 10* np.log10(popt[0])
    B = popt[1] * 180 / np.pi
    C = 10* np.log10(popt[2])
    E = 10* np.log10(popt[3])
    F = popt[4]* 180 / np.pi
    results.append([data.iloc[0]['X'], data.iloc[0]['Y'], A,B,C,E,F])

# Create new results dataframe from the list. Columns=.. names the columns. 
ARA_Results = pd.DataFrame(results, columns = ['X', 'Y', 'A', 'B', 'C', 'E', 'F'])
```

We can look at the mean of the entries.

```text
ARA_Results.mean()
```

The results are broadly comparable with published values, having in mind that our data is not calibrated. We can make a crude plot showing the distribution of the values. They roughly correspond with what we observed in the backscatter mosaic. You can try the other parameters as well.

```text
ylim = (ARA_Results.Y.min(), ARA_Results.Y.max())
ARA_Results.plot.scatter(x='X', y='Y', c='A', ylim = ylim)
```

![](../.gitbook/assets/image%20%283%29.png)

We now have a somewhat massive dataframe of GSAB parameters. The data is difficult to interpret manually. For the time being, we will save the data as a csv file and come back to it later.

```text
ARA_Results.to_csv('GSAB_results_200kHz.txt', sep='\t', index = False)
```

Exit the jupyter console by typing quit\(\)

### Analysis of data with unsupervised machine learning \(kmeans\)

Unsupervised classification groups data into classes without outside ground truthing \(that must later be applied to actually try to understand the classes\). A very common example of unsupervised classification is the kmeans clustering algorithm.

After starting a fresh instance of the ipython qtconsole from cd ~/ecomap\_summerschool/mbsystem\_backscatter/200  
we load the ARA dataset we just created and create a quick plot showing the relationship between the ARA parameters and their change along the profile:

### Load modules and inline plotting

%pylab inline import pandas as pd import numpy as np import sys from sklearn.cluster import KMeans import matplotlib.pyplot as plt import seaborn as sns sys.path.append\('/home/ecomap/ecomap\_summerschool/python\_functions'\) import ecomap\_summerschool\_functions as eco

### Load data

df = pd.read\_csv\('GSAB\_results\_200kHz.txt', sep = '\t'\)

### Have a look at the clustering

sns.pairplot\(df\)

![](../.gitbook/assets/image%20%281%29.png)

We already observe several interesting features, with ‘Y’ being essentially a cross section along the a profile. Especially parameters B, C, E and F clearly display a similar spatial distribution as the higher backscatter intensities we observed in the central part of the backscatter grid. Correspondingly, the histograms of B, C, E and F show a bimodal distribution, despite the data being noisy. We limit ourselves here to the clustering of 2 variables, so we are able to easily display the results. Also, our bounds were not accurate, since we clipped several data values. It would need to be decided whether we clipped real features, or if these are caused by outliers. Let us pick B and C, for example \(and not the maximum backscatter A that would be expected to closely related to the backscatter map\).

We calculate a kmeans classification with three classes on the dataframe columns ‘C’ and ‘B’.

```text
km_3 = KMeans(n_clusters = 3, random_state=42).fit_predict(ARA_Results[['C','B']]) # the [[]] give an array view of these specified columns
# PLot the results
plt.scatter(ARA_Results.C, ARA_Results.B, c=km_3, cmap='tab10')
```

The kmeans algorithm generally picked out the clusters we would have assigned by eye.

![](../.gitbook/assets/image%20%2811%29.png)

Lets do it with more classes

```text
clusters = [2,3,4,5,6,7]
for n in clusters:
    print(' Number of classes: ', n )
    km = KMeans(n_clusters = n, random_state=42).fit_predict(ARA_Results[['E','B']])
    plt.scatter(ARA_Results.E, ARA_Results.B, c=km, cmap='tab10')
    plt.show()
```

We decide to continue with 3 classes…. km = KMeans\(n\_clusters = 3, random\_state=42\).fit\_predict\(ARA\_Results\[\['C','B'\]\]\)

If you look at the km variable, you will note that it contains the labels of the three classes \(0-2\), and has the same lengths as our input dataframe ARA\_Results. Thus, it is easy to add another column to the ARA\_Results dataframe.

```text
ARA_Results['kmeans'] = km
```

Since the order of the data was not changed, we now have a georeferenced dataset of the kmeans classification that we can plot quickly \(or export for gridding and further analysis\).

It is difficult to decide on the number of classes. Obviously, 2 is not enough and 7 is too much, but argument could be made for a few in between \(especially as we have not checked for noisy and clipped data, and so on\). The numbers of classes should be informed both by expert knowledge, as well as analytical techniques such as silouette analysis. The scikit-lean homepage hosts great examples for a number of machine learning topics, including sihouette analysis if you are interested \([http://scikit-learn.org/stable/auto\_examples/cluster/plot\_kmeans\_silhouette\_analysis.html](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)\). In addition, Kmeans often has problems with elongated data blobs \(which we have\), when data are overlapping or there is a lot of noise/outliers. In these cases, Gaussian Mixture is another unsupervised classification that could be tried \([http://scikit-learn.org/stable/modules/mixture.html](http://scikit-learn.org/stable/modules/mixture.html)\).

```text
ylim = (ARA_Results.Y.min(), ARA_Results.Y.max())
ARA_Results.plot.scatter(x='X', y='Y', c='kmeans', cmap='tab10', ylim = ylim)
```

![](../.gitbook/assets/image%20%2813%29.png)

We can compare the rough plot with the backscatter plot we made with mbsystem. Is there any correlation of class 2 with backscatter data? What areas would be good to further investigate with regards to differences in ARA analysis and backscatter strengths?

As a side note, generally the results of machine learning will improve \(greatly\) if data is preprocessed \(outliers, normalization\). A starting point of the options available in scikit learn is available here: [http://scikit-learn.org/stable/modules/preprocessing.html](http://scikit-learn.org/stable/modules/preprocessing.html)

