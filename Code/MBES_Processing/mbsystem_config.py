##############################################################
# LEVEL 1: IMPORT AND BASIC CORRECTIONS
##############################################################
# Control which levels are worked on
LEVEL1 = 'yes'
LEVEL2 = 'yes'
LEVEL3 = 'yes'

####IMPORTANT NOTE
# To allow for the handling of the mb-system scheme for marking processed files
# survey lines MUST NOT END with the letter "p"
####IMPORTANT NOTE

remove_lock_files = 'yes' #Yes tries to remove lockfiles for all files linked in the datalists via mblist
PFAD = '/Volumes/Work/pingo/700/'   # end with /
rekursive_directory_search = 'no'
PREPROCESS = 'no'
FORMAT = 89  # .ALL UND .S7K FILES  work
file_end = '.s7k'
SS_FORMAT = 'C'  # scbw  s snippet c calib. snippet b widebeambackscatter w calibwidebeambackscatter "auto" - no option

AREA = '12.1053/12.11422/54.1844/55.189965'  # WESN. printed in datalist.info at the end of level1
GENERATE_DATALIST = 'yes'
AUTO_CLEAN_BATHY = 'yes'
auto_clean_with_area_boundaries = 'yes'
ATTITUDE_LAG = ''
SELECT_SVP = ''          # mbsvpselect crashing at the moment why?? -> mbsystem bug? has to be done manually atm
SVP = ''             #this is the manual file included in all par files
CORRECT_HPR=''
ROLL_CORR = 0.00
PITCH_CORR = 0.00
CORRECT_TIDE = ''        #no: removes entries from par fileand reprocesses
TIDEFILE = ''  #Tidemode set to 2
CORRECT_DRAFT = ''
DRAFT_CORR = 0.4

EXPORT_NAV = 'no'           # Export Navigation information and stores under profile file name
EXPORT_INFO_LEVEL1 = 'no'   # write output of mbinfo to datalist.info (.e.g, for region boundaries)
##############################################################
# LEVEL 2: Correct Backscatter Data
##############################################################
EXPORT_ARC_CURVES = 'no'
PROCESS_SCATTER = 'yes'  # yes is running mbbackangle
CONSIDER_SEAFLOOR_SLOPE = ''
AVERAGE_ANGLE_CORR = 'no' # backangle correction file specific (no) or average (yes) for complete datesaet

SSS_ACROSS_CUT = 'yes'
SSS_ACROSS_CUT_MIN = -55 #in m, negative is portside
SSS_ACROSS_CUT_MAX = 55 #in m

SSS_CORRECTIONS = 'no' #applies all of the follwoing settings
SSSWATHWIDTH = 160  #that is supposed to e an agnle. I have no clue what happens
# but settings this to any value removes the beams where the roll claib failed...
SSINTERPOLATE = 0
EXPORT_INFO_LEVEL2 = 'no' ## write output of mbinfo to datalist.info (.e.g, for region boundaries)
##############################################################
# LEVEL 3: Make grid data
##############################################################

SCATTER_FILTER = 'low'       #low or high - high not implemented atm works on p-files
INTERPOLATION = '-C3/1'      #up to three cells are interpolated
## Grids
WORK_ON_PER_FILE_BASIS = 'yes'  # Make grids i.e. for each file individually

# Work for both on a per-survey and per file setting
GENERATE_BATHY_GRIDS = 'yes'
GENERATE_SCATTER_GRIDS = 'yes'
SCATTER_WITH_FILTER ='yes'   #Export filtered grids
EXPORT_XYI_FROM_GRID = 'no'
"""
Idea: export first pings with mblist with depth and ship speed and make an educated guess on grid size for each file
"""
BATHY_RES = '-E1/1'
SCATTER_RES = '-E0.25/0.25'

# convert Grids. to implemented atm the results files are converted for speed reasons
UTM_Convert = 'no'
ZONE = '-JU'   # in syntax for mbsystem not implemented in mbsystem atm because of using geographical coordinates throughout

#Only for work on a per-file bases
EXPORT_BEAM_ANGLE = 'no'    #
EXPORT_XYI = 'no'           #
EXPORT_XYZ = 'no'
KFAKTOR = 50               # jeder wievielte Schuss soll exportiert werden mit EXPORT_XYI

FORCE_MBPROCESS = 'yes'   #Force a mbprceoss run; only needed for manual changes
number_of_exceptions = 0
