---
description: 'Author: Peter Feldens'
---

# Creation of Backscatter mosaics

## `Processing of a single line`

Our first task is to process the backscatter information stored within modern multibeam echo sounder datafiles. For this task we will use mbsystem \(Caress and Chayes, 1995; [https://www.mbari.org/products/research-software/mb-system/](https://www.mbari.org/products/research-software/mb-system/)\).

Locate the /home/ecomap/ecomap\_summerschool/mbsystem\_backscatter/200 folder. Inside, a .HSX data file recorded in the North Sea using a Norbit multibeam echo sounder is located. We prepare the file for use in mbsystem and generate datalists referencing the individual data files, similar to the process used during processing of bathymetric data.

```bash
cd /home/ecomap/ecomap_summerschool/mbsystem_backscatter/200
ls -1 *.HSX > temp
mbhysweeppreprocess -F-1 -Itemp -JUTM32N
ls -1 *.mb201 > temp
mbdatalist -F-1 -Itemp > datalist.mb-1
mbdatalist -Z
```

Mbsystem handles three types of swath mapping data: beam bathymetry, beam amplitude, and sidescan. Both amplitude and sidescan represent measures of backscatter strength. Beam amplitudes are backscatter values associated with the same preformed beams used to obtain bathymetry; mbsystem assumes that a bathymetry value exists for each amplitude value and uses the bathymetry beam location for the amplitude. Sidescan is generally constructed with a higher spatial resolution than bathymetry, and carries its own location parameters. The backscatter time series from an individual beam is being commonly referred to as a Snippet. Snippets \(as well as water column data\) are not directly supported in mbsystem.

To have an idea of the data, we plot an image of the raw side scan data. We will not consider the ampltiude data stored with each beam here, normally the side scan data is better. We use the mbm\_plot command, which creates a .cmd file including mbsystem and GMT code to prepare our figure. When an mbsystem command \(normally all commands starting with mbm\) requires the subsequent execution of a .cmd file, please do so, this will not be explicitly stated from now on.



```text
mbm_plot -F-1 -Idatalist.mb-1 -G5 -Osidescan_raw -S


>Plot generation shellscript <sidescan_raw.cmd> created.
>
>Instructions:
>  Execute <sidescan_raw.cmd> to generate Postscript plot <sidescan_raw.ps>.
>  Executing <sidescan_raw.cmd> also invokes gv to view the plot on the screen.

./sidescan_raw.cmd     #This command will not be repeated in the future
```

![](../.gitbook/assets/image.png)

It can clearly be observed that the the angular relationship has not been accurately removed and there is little contrast. It is possible to visualize and correct the angular relationship by mbbackangle, the program used to process backscatter data within mbsystem. Mbbackangle forces a flat incidence angle/intensity relationship, with intensities normalized to a reference angle. In this case we consider 121 angle intervals \(an odd number, so the correction is symmetric around 0\), ranging from -82° to 82° incidence angle, with 40° used as a reference angle. You can have a look at the manpage to figure out what the options of mbbackangle do exactly \(man mbbackangle\) and how the correction is applied.

```text
mbbackangle -Idatalist.mb-1  -V -N121/82.0 -R40
mbset -PSSCORRTYPE:1 #Tell mbsystem there are linear units inside. Only for HSX files. 
mbset -PSSCORRFILE:datalist.mb-1_tot.sga    #Use average for correction
mbset -PDATACUT:2:2:-50:-32  
mbset -PDATACUT:2:2:32:50
```

We use mbset to apply the same correction for all datafiles, instead of doing the angular corrections on a per-file basis: The file datalist.mb-1\_tot.sga contains average correction tables over the complete dataset. You can open it with an editor if you want.

`nano datalist.mb-1_tot.sga`

You can exit the editor with Strg-X. Nano is a very handy command line editor.

The last two lines tell mbsystem that data are stored in linear units within the HSX files. Finally, we remove data of the outer beams with the two calls of mbset with the DATACUT parameter \(this is not generally necessary, but due to some problems with this particular dataset\).

It is possible to run mbbackangle with the -G option on the processed and unprocessed datafiles, to create postscript files showing the impact mbbackangle had on the amplitude/grazing angle relationship. If you want to to try that, refer to the man-page of mbbackangle. We will create our own amplitude/grazing angle curves using Python.

Following the angular corrections, the backscatter data may be filtered. A low pass filter, for example

`mbfilter -Idatalistp.mb-1 -S1/3/3/1`

may considerably reduce „speckle“ noise. Filtered data are plotted by appending „F“ to the data selection option in mbm\_plot or mbm\_grid \(e.g., -G4F for the filtered data\). High-pass filters would be specified using the –D option, and contrast-enhancing filters using the –C option. \(The highpass filter takes a while to run\). Note that this processing command is special because it is applied to the processed datalist and does not require mbprocess to apply!

We can plot the processed data

`mbm_plot -F-1 -Idatalistp.mb-1 -G5F -Osidescan_proc -S`

![](../.gitbook/assets/image%20%2810%29.png)

Finally, similar to bathymetric, backscsatter data can be gridded mbm\_grid, which creates grid files of the backscatter data which is often more convenient for further processing than image formats. The -E option control the grids resolution in meters. -A4F specifies to use filtered side scan data. However, some useful options are better explained on the mbmosaic manpage \(e.g, -A6 - plotting the grazing angle; -Y1 – setting priorities to angles away from the nadir\), so have a look here for the possibilities. Mbgrdviz - a graphical program of mbsystem - can be used to visualize the grids files, but they can also be loaded in GMT scripts, for example.



```text
mbm_grid -F-1 -Idatalistp.mb-1 -E2/2 -A4F -Osidescan_raw_grid
mbgrdviz

```

![](../.gitbook/assets/image%20%286%29.png)

You can create a GeoTif using mbm\_grdtiff, e.g., for loading the data into GIS systems. The simplest option is

`mbm_grdtiff -I<grid_name>`

### Complete backscatter map

Switch to the ~/ecomap\_summerschool/mbsystem\_backscatter/200\_all folder. Here, several 200 kHz lines with strong overlap are stored. The HSX data were already converted to mbsystem, as this was a rather time-intensive procedure.

Using the steps above, create a datalist, make a quick overview map of the side scan data, use mbbackangle to correct the angular relationship and use mbprocess to apply the corrections. Do not forget to use mbset to tell mbsystem data is stored in linear units.

For the following grid operations to assess frequency input, the grids need to have the same extension. The areas extension can be set using mbm\_grid with the -R option. The extents should be Longitude: 07.9572 07.9679 Latitude: 54.7086 54.7300

and the grids should have dimension of 350/1200 grids cells, which results in a resolution of approximately 2 m. We use the -D command of mbm\_grid to specify the extension of the grids in X and Y direction. If multiple grids are to be compared, they need to have identical dimensions. It is also good to try the -Y1 option with mbm\_grid, to give priority to incidence angles away from the nadir. It is useful to write all the command in a script-file \(for example using gedit or nano\) that can be called with

`sh ./<your_script_file>`

Repeat the procedure for the files surveyed with an acoustic frequency of 400 kHz that are stored in ~/ecomap\_summerschool/mbsystem\_backscatter/400\_all

The result should look similar to this \(these are unfiltered grid files\), which is not perfect, but ok for now.

![](../.gitbook/assets/image%20%284%29.png)

### Impact of frequency

We can more directly assess the impact of acoustic frequency on the grids by creating a difference grid. Copy the complete 200 and 400 kHz backscatter grids to ~/ecomap/ecomap\_summerschool and rename the grids to 200.grd and 400.grd, respectively. This can be done using:

```text
cp <old_path/oldname> <newpath/newname>
```

When you have copied and renamed both grids, create a difference grid using GMT \(after navigating the terminal to the folder where the grids are stored\). The grid files need to have the same extents for that operation. You can look at the difference grids using mbgrdviz.

```text
grdmath file1.grd file2.grd SUB=file3.grd
```

How to explain the difference grid? We would need more accurate processing of the data before we can establish frequency differences on this dataset; however, this would be one way to proceed.

### Mbsystem and the Sonar Equation

Currently, there is no way in mbsystem to correct the backscatter data according to the sonar equation. Workarounds involve a\) feeding already calibrated data into mbsystem, or b\) export the data from mbsystem and do the corrections externally. We will do that for the Angular Range Analysis in the next chapter. However, this requires knowledge on absorption, spreading and gain values applied on board and good knowledge on environmental parameters, which is frequently not sufficiently recorded in older datasets \(including our own\).

