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

This information includes everythng we require for the ARC analysis, and we switch to python for the further analysis. You could use a jupyter notebook or jupyter qtconsole to follow the code. We use the jupter qtconsole here. In the following, we will need to apply corrections for absorptiom static survey gains and spreading as well as a footprint correction that is not yet incorporated in the above s7k files (by specification, data in field 7058 should be footprint corrected, but this file is not).

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

### Correct the data and apply the sonar equation

Before we can use the ARA curves for plotting, the data needs to be parameterized. A suitable choice for that purpose is the GSAB model \(Lamarche et al., 2011; [https://www.researchgate.net/publication/222161889\_Quantitative\_characterisation\_of\_seafloor\_substrate\_and\_bedforms\_using\_advanced\_processing\_of\_multibeam\_backscatter-Application\_to\_Cook\_Strait\_New\_Zealand](https://www.researchgate.net/publication/222161889_Quantitative_characterisation_of_seafloor_substrate_and_bedforms_using_advanced_processing_of_multibeam_backscatter-Application_to_Cook_Strait_New_Zealand)\)

Since we already have the ARA curves in a pandas dataframe, application of the model is comparatively straightforward. The tasks involve fitting the raw data to a curve function given by the GSAB model and then calculating the parameters A to F. This needs to be done for every or a small number of pings, otherwise we would calculate the average over larger areas. Fortunately, we already exported a PingCount variable with mblist and loaded it into the dataframe. Also, we need to convert the linear intensity data to decibels.

For the next operations, we prepared some custom functions (note: these were written for traching purposes, please do not use in actual work without checking for errors).
```text

def convert_to_dB(dB_column, SL=100):
    return 20*np.log10(dB_column) - SL

def correct_spreading_offset_absorption(slant, spreading, absorption):
    TL = spreading * np.log10(slant) + 2 * slant * absorption * 0.001
    return TL

```

Now we convert the linear amplitude to dB values, which we store in a new column of the dataframe. We get a lot of -inf values due to many 0-intensity data points. We will remove these points as well as missing datapoints from our dataset. Take a moment to understand the logic and syntax in the command.

```python
#Make new column in dataframe for dB values
df_raw['BS_dB'] = convert_to_dB(df_raw['BS'])

#Remove the -infs caused by log(0) with 0
df_raw.drop(df_raw[df_raw.isin([np.nan, np.inf, -np.inf]).any(1)].index, inplace = True)
```

Calibrated high-frequency backscatter data typically shows intensities between -70 and 0 dB depending on angle and seafloor. Our data here is uncalibrated and should be corrected according to the sonar equation as far as possible. The \(incomplete\) sonar equation is given with: TS = EL - SL + 2\*TL with TS: target strength, EL: echo level, SL: source level, and TL: transmission loss. The transmission loss TL is:

TL = Spreading + Absorption

For brevity, we ignore the effects of the insonifed area; details for the computation could be found in \(Lamarche and Lurton, 2015\). Prior to computing the sonar equation, the gains applied during the survey have to be removed to obtain the echo level EL. Fortunately, for this survey everything was correctly recorded and no changes are required. If so, the difference to the values recorded during the survey needs to be used. 

We correct the dB intensity values in the dataframe. In addition, we calculate the required slant range based on transducer altitude and across-track range \(ignoring along-track components\) that we exported with mblist.

```python
spreading = 0
absorption = 0  # dB/km
static = 0

# Get slant range in m
df_raw['Slant'] = np.sqrt(Height**2 + df_raw.Across_Dist**2)

# Correct values
df_raw['TS'] = correct_spreading_offset_absorption(slant, spreading, absorption)

#Plot the first 20000 data points
df_raw.iloc[0:20000].plot.scatter(x='Angle', y='TS')
```

To account for the insonified area, we follow equations given by Malik 2019. The paper is freely available online https://www.mdpi.com/2076-3263/9/4/183, and explains the reasoning behing the equations. 

```python
def footprint_calc(slant, height, c, pulse, angle, theta_across, theta_along):
    theta_across = np.deg2rad(theta_across)
    theta_along = np.deg2rad(theta_along)
    angle = np.deg2rad(angle)
    A1 = theta_across * theta_along * slant**2 * 1 / (np.cos(angle))
    A2 = theta_along * slant * c * pulse * 1 / (2*np.sin(angle))
    return np.min([A1, A2])

pulse_length = 0.0002
theta_across = 0.9 #across-track beamwidth of the multibeam system in degrees
theta_along= 0.9 #along-track beamwidth of the multibeam system in degrees
 
df['Footprint'] = df.apply(lambda row: footprint_calc(row['Slant'], row['Height'], c, pulse_length, row['Angle'], theta_across, theta_along), axis=1)
```

We have a very high angle resolution that we do not need and which will slow done the ARA analysis massively. Thus, lets bin the data into 2° and 50 ping intervals to reduce scatter. In addition, we limit the data to incidence angles larger than 0° \(thus looking only at the starboard part of the swath width\).

```python
#Group data by Ping Counts and bin into 2° intervals, stacking over 30 pings 
bins_angle = np.arange(-80,80,2) 
ping_bin_number = 50
bins_ping = np.arange(df_raw.Ping.min()-ping_bin_number,df_raw.Ping.max()  +   ping_bin_number,ping_bin_number) 

## Binning Angles
# Sort incidence angles indices into bins_angle
bins_angle_idx = np.digitize(df_raw.Angle, bins_angle)
# Make new dataframe columns
data = []  
for i in range(bins_angle_idx.size):
     data.append(bins_angle[bins_angle_idx[i]])   #Right boundary, not precise
df_raw['Angle_binned'] = data

## Binning Pings
bins_ping_idx = np.digitize(df_raw.Ping, bins_ping)
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

These results can now be used for various applications of Angular Response Analysis approaches. One example of this is the mentioned GSAM model. 


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
        
def calculate_GSAB(index, df_bins, bounds_min, bounds_max, arc):
    arc = []
    ping_data = df_bins[df_bins['Pings_binned'] == index]  # extract one ping
    count = ping_data.Corr_BS.count()
    if count > 20:   # we want at least 20 reasonable values
        w, _ = opt.curve_fit(gsab_func, ping_data.Angle_binned, ping_data.Corr_BS,
                             p0=x0, maxfev=50000, bounds=(bounds_min, bounds_max))

        arc.append([w[0], w[1], w[2], w[3], w[4], index, count])

    cleanedArc = [x for x in arc if str(x) != 'None']
    return cleanedArc

# Split data into individual pings
ping_numbers = df_bins['Pings_binned'].unique()
gsab = []
# set bounds for A-F to speed up fitting. Roughly based on values reported by Lamarche et al (2011).
bounds_min = [10**(-35/10), 0.1*np.pi/180, 10**(-30/10), 10**(-30/10), 5*np.pi/180]
bounds_max = [10**(25/10), 35*np.pi/180, 10**(5/10), 10**(20/10), 25*np.pi/180]

# Iterate over individual pings. This could be vectorized and be much faster for 
# application on real datasets. The loop may take a moment.
gsab = calculate_GSAB)(index, df_bins, bounds_min, bounds_max, gsab)
                                        for index in tqdm(ping_numbers)
gsab = [item for sublist in gsab for item in sublist]  # flatten list

# Create new results dataframe from the list. Columns=.. names the columns. 
gsab_df = pd.DataFrame(gsab, columns = ["A", "B", "C", "E", "F", "Ping", "Count"])
```

We can look at the mean of the entries.

```text
gsab_df.mean()
```

The results are broadly comparable with published values, having in mind that our data is not calibrated and has not been properly processed. We can make a crude plot showing the distribution of the values. They roughly correspond with what we observed in the backscatter mosaic. You can try the other parameters as well.

```text
ylim = (gsab_df.Y.min(), gsab_df.Y.max())
gsab_df.plot.scatter(x='X', y='Y', c='A', ylim = ylim)
```

![](./GSAB_example.png)

We now have a somewhat massive dataframe of GSAB parameters. The data is difficult to interpret manually. For the time being, we will save the data as a csv file and come back to it later.

```python
gsab_df.to_csv('GSAB_results_400kHz.txt', sep='\t', index = False)
```

Exit the jupyter console by typing quit\(\)

### Analysis of data with unsupervised machine learning \(kmeans\)

Unsupervised classification groups data into classes without outside ground truthing \(that must later be applied to actually try to understand the classes\). A very common example of unsupervised classification is the kmeans clustering algorithm.

After starting a fresh instance of the ipython qtconsole from  we load the ARA dataset just created and create a quick plot showing the relationship between the ARA parameters and their change along the profile:

### Load modules and inline plotting

```python
%pylab inline 
import pandas as pd 
import numpy as np 
import sys 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
import seaborn as sns 

```

### Load data

df = pd.read\_csv\('GSAB\_results\_400kHz.txt', sep = '\t'\)

### Have a look at the clustering
```python
sns.pairplot\(df\)
```

![](./img/arc_results_clustering.png)

ToDo: Feature description

We calculate a kmeans classification with three classes on the dataframe columns ‘C’ and ‘B’.

```python
km_3 = KMeans(n_clusters = 3, random_state=42).fit_predict(df[['C','B']]) 

# Plot the results
plt.scatter(df.C, df.B, c=km_3, cmap='tab10')
```

The kmeans algorithm generally picked out the clusters we would have assigned by eye.

![](./img/arc_kmeans_plot.png)

Lets do it with more classes

```python
clusters = [2,3,4,5,6,7]
for n in clusters:
    print(' Number of classes: ', n )
    km = KMeans(n_clusters = n, random_state=42).fit_predict(df[['E','B']])
    plt.scatter(df.E, df.B, c=km, cmap='tab10')
    plt.show()
```

We decide to continue with 3 classes…. 
```python
km = KMeans(n_clusters = 3, random_state=42).fit_predict(df[['C','B']])
```

If you look at the km variable, you will note that it contains the labels of the three classes \(0-2\), and has the same lengths as our input dataframe with the results. Thus, it is easy to add another column to the ARA\_Results dataframe.

```text
df['kmeans'] = km
```

Since the order of the data was not changed, we now have a georeferenced dataset of the kmeans classification that we can plot quickly \(or export for gridding and further analysis\).

It is difficult to decide on the number of classes. Obviously, 2 is not enough and 7 is too much, but argument could be made for a few in between \(especially as we have not checked for noisy and clipped data, and so on\). The numbers of classes should be informed both by expert knowledge, as well as analytical techniques such as silouette analysis. The scikit-lean homepage hosts great examples for a number of machine learning topics, including sihouette analysis if you are interested \([http://scikit-learn.org/stable/auto\_examples/cluster/plot\_kmeans\_silhouette\_analysis.html](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)\). In addition, Kmeans often has problems with elongated data blobs \(which we have\), when data are overlapping or there is a lot of noise/outliers. In these cases, Gaussian Mixture is another unsupervised classification that could be tried \([http://scikit-learn.org/stable/modules/mixture.html](http://scikit-learn.org/stable/modules/mixture.html)\).

```text
ylim = (df.Y.min(), df.Y.max())
df.plot.scatter(x='X', y='Y', c='kmeans', cmap='tab10', ylim = ylim)
```

![](./img/arc_results_of_gsab_kmeans.png)

We can compare the rough plot with the backscatter plot we made with mbsystem. Is there any correlation of class 2 with backscatter data? What areas would be good to further investigate with regards to differences in ARA analysis and backscatter strengths?

As a side note, generally the results of machine learning will improve \(greatly\) if data is preprocessed \(outliers, normalization\). A starting point of the options available in scikit learn is available here: [http://scikit-learn.org/stable/modules/preprocessing.html](http://scikit-learn.org/stable/modules/preprocessing.html)

